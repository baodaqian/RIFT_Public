#!/usr/bin/env python3
"""PVC (Intel XPU) frontend for the GOTCHA combined dataset.

Copy of ``train_gotcha_dataset.py`` (unchanged on disk) with the backend table
pointing at the ``_pvc`` entry points: built-in RIFT/MFBP methods dispatch to
``rift_pvc.gotcha_training``; baselines are discovered from ``train_<x>_pvc.py``
through the same literal ``GOTCHA_BACKEND`` declaration and ``run_gotcha``
callable contract as the CUDA frontend. Default device is ``xpu``. Planning,
selection, roles and output layout are identical; the built-in recipe adds
``sector_execution`` (``--sector-execution``, default batched).

Original docstring:
Root adapter for joint-pass GOTCHA training on named regions.

Defaults: Camry, passes 1..8, HH, 1500/440/440 pass-sector viewpoints.
--dry-run performs metadata-only planning. Actual fitting belongs inside an
experiment-manager allocation. Baselines opt in to the native acquisition
contract; their existing B787 entrypoints are never repurposed implicitly.
"""
from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import json
import math
import os
from pathlib import Path

from rift.gotcha_dataset import DEFAULT_ROOT, PROJECT_ROOT, DEFAULT_NUM_TRAIN, GOTCHADataset, load_region, selection, training_sectors

METHODS = ('rift', 'rift_grid', 'isotropic', 'spinr', 'radar_fields', 'geraf',
           'radarsplat', 'sugavanam_ertin', 'sh_sas', 'fsh', 'mfbp')
BUILTIN = ('rift', 'rift_grid', 'isotropic', 'mfbp')
ALIASES = {'adaptive': 'rift', 'adaptive_rift': 'rift', 'grid_sh': 'rift_grid',
           'grid': 'isotropic', 'sp0': 'isotropic', 'rf': 'radar_fields',
           'se': 'sugavanam_ertin', 'rs': 'radarsplat'}
BASELINE_MODULES = {
    'spinr':'train_spinr_style_pvc', 'radar_fields':'train_radar_fields_pvc',
    'geraf':'train_geraf_pvc', 'radarsplat':'train_radarsplat_pvc',
    'sugavanam_ertin':'train_sugavanam_ertin_pvc', 'sh_sas':'train_sh_sas_pvc',
    'fsh':'scripts_pvc.eval_rift_dataset_model_free_pvc',
}
BUILTIN_MODULE = 'rift_pvc.gotcha_training'
BACKEND_SCHEMA = 'rift_gotcha_backend_v1'


def _read_backend_spec(path, module, method):
    tree = ast.parse(path.read_text(), filename=str(path))
    spec, declared = None, False
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        if any(isinstance(t, ast.Name) and t.id == 'GOTCHA_BACKEND' for t in targets):
            declared = True
            try:
                spec = ast.literal_eval(node.value)
            except (ValueError, TypeError) as exc:
                raise ValueError(f'{module}.GOTCHA_BACKEND must be a literal mapping') from exc
    if not declared:
        return None
    required = dict(schema=BACKEND_SCHEMA, method=method,
                    selection_unit='pass_sector', joint_passes=True,
                    native_frequency_policy='ragged_exact')
    if not isinstance(spec, dict) or any(spec.get(k) != v for k, v in required.items()):
        raise ValueError(f'{module}: incompatible GOTCHA_BACKEND capability declaration')
    if not isinstance(spec.get('polarizations'), list) or not spec['polarizations'] or not set(spec['polarizations']) <= {'hh', 'hv', 'vh', 'vv'}:
        raise ValueError(f'{module}: invalid supported polarizations')
    if not isinstance(spec.get('metric_domain'), str) or not spec['metric_domain']:
        raise ValueError(f'{module}: native metric domain must be declared')
    function = spec.get('callable', 'run_gotcha')
    if not isinstance(function, str) or not any(isinstance(n, ast.FunctionDef) and n.name == function for n in tree.body):
        raise ValueError(f'{module}: declared GOTCHA callable is missing')
    return spec


def backend_registry():
    """Read literal capability declarations without importing baseline code.

    A baseline owner adds GOTCHA_BACKEND to its root module and the named
    callable. This socket allows concurrent implementations to integrate
    without modifying this entrypoint or weakening legacy contracts.
    """
    result = {m:dict(status='available', module=BUILTIN_MODULE, builtin=True,
                     polarizations=['hh','hv','vh','vv']) for m in BUILTIN}
    for method, module in BASELINE_MODULES.items():
        path = PROJECT_ROOT / (module.replace('.', '/') + '.py')
        if path.is_file():
            try:
                spec = _read_backend_spec(path, module, method)
            except (OSError, SyntaxError, ValueError, TypeError) as exc:
                # Invalid unselected modules cannot block another method. All
                # capability checks remain mandatory when this method is selected.
                result[method] = dict(status='invalid', module=module, builtin=False,
                                      reason=f'{module}: {exc}')
                continue
            if spec is not None:
                result[method] = dict(spec, status='available', module=module, builtin=False)
                continue
        result[method] = dict(status='unavailable', module=module, builtin=False,
            reason='No PVC entry point with an explicit native GOTCHA backend yet (see RIFT_PVC_Adaptation.md).')
    return {method:result[method] for method in METHODS}


def resolve_methods(requested, registry, polarizations):
    requested = list(requested)
    names = [m for m in METHODS if registry[m]['status'] == 'available' and set(polarizations) <= set(registry[m].get('polarizations', []))] if requested == ['all'] else [ALIASES.get(m,m) for m in requested]
    if not names or len(names) != len(set(names)):
        raise ValueError('Select at least one distinct compatible method')
    for method in names:
        if method not in registry:
            raise ValueError(f'Unknown method {method!r}')
        if registry[method]['status'] != 'available':
            raise ValueError(f'{method}: {registry[method]["reason"]}')  # 'invalid' declarations raise here
        if not set(polarizations) <= set(registry[method]['polarizations']):
            raise ValueError(f'{method} does not support the selected polarization heads')
    return names


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    from rift.antenna_selection import add_arguments
    add_arguments(p)
    p.add_argument('--dataset-root', type=Path, default=DEFAULT_ROOT)
    p.add_argument('--shard-root', type=Path, help='Override the native raw-storage NPZ directory (default DATASET_ROOT/New_Transfer/shards)')
    p.add_argument('--region', default='camry')
    p.add_argument('--region-config', type=Path, help='Additional named placements, schema rift_gotcha_regions_v1')
    p.add_argument('--passes', nargs='+', type=int, default=list(range(1,9)))
    p.add_argument('--polarizations', nargs='+', type=str.lower, default=['hh'])
    p.add_argument('--num-train', type=int,
                   help='Total training pass-sectors, balanced across selected passes; default min(1500, 250 × passes). Holdouts stay fixed.')
    p.add_argument('--method', nargs='+', default=['rift'], help='Method selectors; all runs every explicitly compatible backend')
    configs = p.add_mutually_exclusive_group()
    configs.add_argument('--method-config', type=Path, help='JSON mapping selected canonical method names to backend-specific configuration objects')
    configs.add_argument('--config', type=Path, help='Backend configuration JSON for exactly one baseline (not RIFT/MFBP)')
    p.add_argument('--output-root', type=Path, default=PROJECT_ROOT/'training_checkpoints'/'GOTCHA_dataset')
    p.add_argument('--resume', type=Path, help='One method only; requires this adapter\'s exact source/region/split/recipe')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--check-initialization', action='store_true', help='SE only: bounded CPU initialization diagnostic; no fitting')
    p.add_argument('--se-stage1-only', action='store_true',
                   help='SE only: stop after sparse Stage 1 with the full recipe retained for a dependent SDF job')
    p.add_argument('--list', action='store_true', help='List methods and their GOTCHA readiness without opening the dataset')
    p.add_argument('--device', default='xpu')
    # New GOTCHA engineering recipe. These are not an inherited production
    # budget or an assertion of equivalence to the synthetic collection preset.
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--granularity', type=int, default=48)
    p.add_argument('--max-points', type=int, default=262144)
    p.add_argument('--sh-degree', type=int, default=3)
    p.add_argument('--sh-init-degree', type=int, default=0,
                   help='Adaptive RIFT: SH bands 0..N trainable from the first update (default 0; tuning campaign B7)')
    p.add_argument('--support-box', type=float, nargs=3, metavar=('HX', 'HY', 'HZ'),
                   help='Adaptive RIFT: box support centred at the region origin (half-extents, m) instead of the cube')
    p.add_argument('--support-pitch', type=float, default=0.125, help='Cubic cell pitch of --support-box (m)')
    # Sparsity levers (tuning campaign A31/B29): train.py's own --prune-* target mode and a multiple of the
    # dimensionless group-L1 strength mu1. Opt-in; absent from default recipes.
    p.add_argument('--prune-every', type=int, default=0,
                   help="Adaptive RIFT, B787 optimizer: train.py's pruning every N epochs (0 disables; --prune-mode target)")
    p.add_argument('--prune-target-active', type=int, default=0,
                   help="train.py --prune-target-active: active points to anneal down to on the cubic ramp, then held")
    p.add_argument('--prune-start-epoch', type=int, default=0, help='train.py --prune-start-epoch (1-based; 0 = first --prune-every hit)')
    p.add_argument('--prune-end-epoch', type=int, default=0, help='train.py --prune-end-epoch (1-based; 0 = --epochs)')
    p.add_argument('--prune-min-active', type=int, default=0, help='train.py --prune-min-active floor')
    p.add_argument('--step-every', type=int, default=1,
                   help="Adaptive RIFT: train.py's --step-every N, one optimizer step per N pass-sectors with their "
                        "gradients summed (default 1); pair with lr/N to keep the step's stability number")
    p.add_argument('--mu1-scale', type=float, default=1.0,
                   help='Adaptive RIFT with RIFT-dataset priors: multiply the dimensionless group-L1 strength mu1 (and so l1_weight) by this factor')
    # Interpolation split and trilinear densification (tuning campaign A41). Opt-in; absent from default recipes.
    p.add_argument('--unit-split-stride', type=int, default=0,
                   help='Pass-held-out unit split (rift_pvc/gotcha_unit_split.py): keep every Nth unsealed sector ID in '
                        'all passes; 0 keeps the registered sector split')
    p.add_argument('--unit-split-heldout-pass', type=int, default=4, help='Unit split: the pass whose units are held out')
    p.add_argument('--unit-split-heldout-fraction', type=float, default=0.5,
                   help='Unit split: seeded fraction of the selected IDs held out in that pass')
    p.add_argument('--densify-epochs', type=int, nargs='+', default=[],
                   help='Adaptive RIFT, B787 optimizer: Plenoxel-style trilinear densify events after these 1-based '
                        'epochs (rift_pvc/gotcha_densify.py); each halves the lattice pitch')
    p.add_argument('--densify-max-active', type=int, default=0, help='Densify: active-point cap (default --max-points)')
    p.add_argument('--densify-energy-floor', type=float, default=1e-3,
                   help='Densify: drop points below this fraction of the largest point energy before an event')
    p.add_argument('--densify-init', choices=('trilinear', 'inherit'), default='trilinear',
                   help="Densify: children start from the trilinear interpolation of the parent lattice (Plenoxels; "
                        "default) or inherit the parent (heir at its exact position, seven zero siblings; render-preserving)")
    p.add_argument('--densify-normalize', choices=('energy', 'volume'), default='volume',
                   help="Densify: 'volume' multiplies the interpolated children by --densify-weight-scale (default, A41); "
                        "'energy' rescales them to the kept parents' coefficient energy (B52: leaves a null-space component)")
    p.add_argument('--densify-weight-scale', type=float, default=0.125,
                   help="Densify, --densify-normalize volume: factor on the interpolated child weight (1/8, the child volume share)")
    p.add_argument('--densify-lr-rule', choices=('fixed', 'curvature'), default='curvature',
                   help="Densify: after each event, 'curvature' measures the per-update Hessian's lambda_max at the current "
                        "gain and lowers every rate so (base lr / eps) lambda_max <= --densify-lr-target (A42); "
                        "'fixed' multiplies every rate by --densify-lr-factor")
    p.add_argument('--densify-lr-target', type=float, default=1.0, help='Densify, curvature rule: stability-number ceiling')
    p.add_argument('--densify-curvature-units', type=int, default=2, help='Densify, curvature rule: TRAIN units measured')
    p.add_argument('--densify-curvature-iterations', type=int, default=10, help='Densify, curvature rule: power iterations')
    p.add_argument('--densify-lr-factor', type=float, default=1.0,
                   help='Densify, fixed rule: multiply every learning rate (and the cosine base and floor) by this after each event')
    p.add_argument('--densify-max-level', type=int, default=3, help='Densify: last lattice level (0.125 m box -> 3 = 1.56 cm)')
    p.add_argument('--densify-curvature-start', action='store_true',
                   help='Densify, curvature rule (opt-in, A62): measure once after the start and set the base rate so '
                        '(base lr / eps) lambda_max = --densify-lr-target (may raise)')
    p.add_argument('--gain-warm-start', choices=('first_pulse', 'pooled'), default='first_pulse',
                   help="Adaptive RIFT, backprojection start (A62): 'pooled' re-fits the warm-started complex gain by "
                        "least squares over every start pulse (default: train.py's first pulse)")
    p.add_argument('--gain-lr', type=float, default=0.0,
                   help='Adaptive RIFT (opt-in, A62): own AdamW group for the gain at this rate, excluded from '
                        'curvature cuts (0: the gain shares the coefficient group, as train.py)')
    p.add_argument('--gain-eps', type=float, default=1e-8, help='Adam eps of the --gain-lr group')
    p.add_argument('--densify-curvature-guard', nargs='*', choices=('growth', 'restart'), default=[],
                   help='Densify, curvature rule (opt-in, A57): re-measure and re-apply the rule after an SH growth event '
                        '(growth) and/or a cosine warm restart (restart) on epochs without a densify event; never raises a rate')
    p.add_argument('--train-eval-every', type=int, default=0,
                   help='Adaptive RIFT: score every TRAIN unit every N epochs (0: at the final epoch only)')
    # Unset optimizer flags take the selected optimizer recipe's values
    # (rift_grid/isotropic always use the earlier GOTCHA values).
    p.add_argument('--optimizer', choices=('b787', 'legacy'), default='b787',
                   help="Adaptive RIFT only: the RIFT-dataset B787 optimizer and schedule (AdamW lr/pos-lr 0.003, eps 1e-8; cosine warm restarts T0 10, Tmult 2 per epoch; a refinement event every 10 epochs at B787's fractions, exposures and split level; one fixed training order) or the earlier constant-LR GOTCHA optimizer and its checkpoints")
    p.add_argument('--lr', type=float, help='Scene and gain learning rate (0.003)')
    p.add_argument('--pos-lr', type=float, help='Position learning rate (B787 0.003; legacy 0.0001)')
    p.add_argument('--adam-eps', type=float, help='AdamW epsilon (B787 1e-8; legacy 1e-20)')
    p.add_argument('--point-chunk', type=int, default=4096)
    p.add_argument('--range-model', choices=('sum2', 'unit'), default='sum2',
                   help='Learned RIFT/grid/isotropic amplitude law: sum2 is the RIFT-dataset operator 1/((4pi)^2((2r)^2+1e-9)) on the physical range; unit reproduces the earlier GOTCHA kernel and its checkpoints. MFBP ignores it.')
    p.add_argument('--initialization', choices=('backprojection', 'random'), default='backprojection',
                   help='Adaptive RIFT only: the RIFT-dataset start (zero scene; coherent backprojection of the first --bp-views pass-sectors of the epoch-1 training order in the ROI-projected loss domain; gain warm start; B787 coefficient gauge) or the earlier random 1e-3 start')
    p.add_argument('--bp-views', type=int, default=100, help='Adaptive RIFT backprojection start: training pass-sectors used (RIFT-dataset recipe: 100 views)')
    p.add_argument('--priors', choices=('rift_dataset', 'none'), default='rift_dataset',
                   help="Adaptive RIFT only: train.py's group-L1 and SH-degree priors at the RIFT-dataset strength relative to the data and the initial scene (mu1, mu2 measured on B787; requires the backprojection start) or none")
    p.add_argument('--pulses-per-sector', type=int, default=0,
                   help='All methods and train/validation/test roles: use one fixed seed-42 subset of up to N native pulses per sector/channel; 0 keeps all. Normalization uses selected TRAIN pulses only.')
    p.add_argument('--refine-every', type=int, help='Adaptive RIFT refinement cadence: epochs under B787 (10), updates under legacy (100)')
    p.add_argument('--probe-every', type=int, help='Adaptive RIFT SH-band probe stride in pass-sectors (B787 16, rotated by the epoch; legacy 10)')
    p.add_argument('--refine-fraction', type=float, help='Legacy only: one spatial/angular refinement fraction (0.05); B787 uses 1/512 and 1/16')
    p.add_argument('--max-level', type=int, help='Maximum spatial split level (B787 1; legacy 3)')
    p.add_argument('--checkpoint-every', type=int, default=10, help='Completed pass-sector optimizer steps between recovery checkpoints')
    p.add_argument('--sector-execution', choices=('batched', 'loop'), default='batched',
                   help='Built-in RIFT backends on PVC: render all selected pulses of a pass-sector in one kernel per point chunk with one backward pass (batched, default; recorded in the recipe as batched_pulses_one_backward_v1) or the original per-pulse loop')
    p.add_argument('--forward-evaluation', choices=('direct', 'nufft'), default='direct',
                   help='Adaptive RIFT on PVC: exact direct summation at the selected native bins (default) or the RIFT-dataset operator\'s type-1 NUFFT of the same kernel (rift_pvc/gotcha_nufft.py; control, recorded as rift_dataset_nufft_v1)')
    p.add_argument('--loss-domain', choices=('roi_projected', 'full_native'), default='roi_projected',
                   help='Adaptive RIFT on PVC: ROI range-subspace projected complex MSE (default) or train.py\'s complex MSE over every selected native bin, normalized by TRAIN mean full-native power (control; checkpoints selected by validation full-native RelMSE)')
    p.add_argument('--frequency-stride', type=int, choices=(1, 2), default=1,
                   help='All methods and train/validation/test roles: fixed native frequency stride retaining both bandwidth endpoints. Test payloads stay sealed; 1 preserves all frequencies.')
    args = p.parse_args(argv)
    for name in ('epochs','granularity','max_points','point_chunk','refine_every','probe_every','checkpoint_every','bp_views'):
        if getattr(args,name) is not None and getattr(args,name) <= 0:
            p.error(f'--{name.replace("_", "-")} must be positive')
    if args.priors == 'rift_dataset' and args.initialization != 'backprojection':
        p.error('--priors rift_dataset is defined at the backprojection start; add --initialization backprojection or --priors none')
    if args.pulses_per_sector < 0:
        p.error('--pulses-per-sector must be nonnegative (0 means all)')
    if args.granularity < 2 or args.max_points < args.granularity**3:
        p.error('granularity must be >=2 and max-points must hold its initial cubic grid')
    if not 0 <= args.sh_init_degree <= args.sh_degree:
        p.error('--sh-init-degree must lie in 0..--sh-degree')
    if args.support_box is not None and (min(args.support_box) <= 0 or args.support_pitch <= 0
                                         or any(abs(round(2 * h / args.support_pitch) * args.support_pitch - 2 * h) > 1e-9
                                                for h in args.support_box)):
        p.error('--support-box half-extents must be positive whole multiples of half --support-pitch')
    if args.prune_every < 0 or args.prune_start_epoch < 0 or args.prune_end_epoch < 0 or args.prune_min_active < 0:
        p.error('--prune-* values must be nonnegative')
    if args.prune_every and args.prune_target_active <= 0:
        p.error('--prune-every uses train.py\'s target mode and needs --prune-target-active > 0')
    if args.prune_every and (args.optimizer != 'b787' or args.initialization != 'backprojection'):
        p.error('--prune-every is implemented for the B787 optimizer at the backprojection start')
    if args.step_every < 1:
        p.error('--step-every must be at least 1')
    if not math.isfinite(args.mu1_scale) or args.mu1_scale <= 0:
        p.error('--mu1-scale must be positive and finite')
    if args.mu1_scale != 1.0 and args.priors != 'rift_dataset':
        p.error('--mu1-scale scales the RIFT-dataset priors; it needs --priors rift_dataset')
    if args.unit_split_stride < 0 or not 0 < args.unit_split_heldout_fraction < 1:
        p.error('--unit-split-stride must be nonnegative and --unit-split-heldout-fraction lie in (0, 1)')
    if args.unit_split_stride and args.unit_split_heldout_pass not in args.passes:
        p.error('--unit-split-heldout-pass must be one of --passes')
    if args.densify_epochs:
        if args.optimizer != 'b787' or args.prune_every:
            p.error('--densify-epochs runs at the B787 epoch end and excludes --prune-every')
        if min(args.densify_epochs) < 1 or len(set(args.densify_epochs)) != len(args.densify_epochs):
            p.error('--densify-epochs must be distinct positive epochs')
        if args.densify_max_active < 0 or args.densify_max_active > args.max_points:
            p.error('--densify-max-active must lie in [0, --max-points]')
        if not (args.densify_energy_floor >= 0 and args.densify_weight_scale > 0 and args.densify_lr_factor > 0
                and math.isfinite(args.densify_weight_scale) and math.isfinite(args.densify_lr_factor)):
            p.error('--densify-energy-floor must be nonnegative, weight scale and lr factor positive and finite')
        if args.densify_max_level < 1:
            p.error('--densify-max-level must be at least 1')
        if not (args.densify_lr_target > 0 and args.densify_curvature_units >= 1 and args.densify_curvature_iterations >= 3):
            p.error('--densify-lr-target must be positive, curvature units >= 1 and iterations >= 3')
        if args.densify_lr_rule == 'curvature' and args.loss_domain != 'full_native':
            p.error('--densify-lr-rule curvature needs --loss-domain full_native')
        if args.densify_curvature_guard and args.densify_lr_rule != 'curvature':
            p.error('--densify-curvature-guard needs --densify-lr-rule curvature')
        if args.densify_curvature_start and args.densify_lr_rule != 'curvature':
            p.error('--densify-curvature-start needs --densify-lr-rule curvature')
    elif args.densify_curvature_guard or args.densify_curvature_start:
        p.error('--densify-curvature-guard/--densify-curvature-start need --densify-epochs')
    if args.gain_warm_start == 'pooled' and args.initialization != 'backprojection':
        p.error('--gain-warm-start pooled re-fits the adaptive-RIFT backprojection start')
    if args.gain_lr < 0 or not (args.gain_eps > 0 and math.isfinite(args.gain_eps) and math.isfinite(args.gain_lr)):
        p.error('--gain-lr must be nonnegative and --gain-eps positive and finite')
    if args.train_eval_every < 0:
        p.error('--train-eval-every must be nonnegative')
    if args.sh_degree < 0 or args.sh_degree > 10 or (args.max_level is not None and args.max_level < 0):
        p.error('SH degree must be 0..10 and max-level nonnegative')
    if args.refine_fraction is not None and (not math.isfinite(args.refine_fraction) or not 0 <= args.refine_fraction <= 1):
        p.error('refine-fraction must be finite in [0,1]')
    if any(getattr(args,k) is not None and (not math.isfinite(getattr(args,k)) or getattr(args,k) <= 0)
           for k in ('lr','pos_lr','adam_eps')):
        p.error('Learning rates and Adam epsilon must be positive and finite')
    if args.seed != 42:
        p.error('The registered parent split and nested subsets fix seed 42')
    try:
        from rift.antenna_selection import from_args
        from_args(args, source=(1, 1), default=1)
        args.passes, args.polarizations = selection(args.passes,args.polarizations)
        if args.num_train is None:
            args.num_train = min(DEFAULT_NUM_TRAIN, 250 * len(args.passes))
        training_sectors(args.passes, args.num_train)
    except ValueError as exc:
        p.error(str(exc))
    return args


def with_isolated_target(dataset, recipe):
    """Name a box-isolated target in the recipe (tuning campaign; reviewer B-note after B12).

    Filtered shards (scripts_pvc/gotcha_filtered_shard.py) carry an ``isolation`` block. With
    --pulses-per-sector 0 the recipe would otherwise report 'all_native' pulses, although the rows
    are the filtered, decimated ones. Shards without the block leave the recipe unchanged.
    """
    blocks = {}
    for (pass_id, pol), shard in dataset.shards.items():
        block = shard.metadata.get('isolation')
        if block is not None:
            blocks[f'pass{pass_id}_{pol}'] = {k: block.get(k) for k in (
                'schema', 'status', 'box', 'operating_point', 'keep_every', 'row_selection', 'stored_response',
                'test_rows', 'git_commit')}
    if not blocks:
        return recipe
    if len(blocks) != len(dataset.shards):
        raise ValueError('Mixed isolated and native shards in one dataset')
    first = next(iter(blocks.values()))
    if any({k: v for k, v in b.items() if k != 'git_commit'} != {k: v for k, v in first.items() if k != 'git_commit'}
           for b in blocks.values()):
        raise ValueError('Shards disagree on the isolation operating point')
    return dict(recipe, target=dict(kind='box_isolated_filtered_shard', **{k: v for k, v in first.items()
                                                                            if k != 'git_commit'},
                                    shard_commits={k: b['git_commit'] for k, b in blocks.items()},
                                    pulse_policy=f"filtered rows, every {first['keep_every']}th native pulse"))


def make_plan(args):
    from rift.antenna_selection import from_args
    from_args(args, source=(1, 1), default=1)
    registry = backend_registry()
    methods = resolve_methods(args.method, registry, args.polarizations)
    if args.resume and (len(methods) != 1 or methods[0] == 'mfbp'):
        raise ValueError('Resume requires one learned method')
    if args.check_initialization and methods != ['sugavanam_ertin']:
        raise ValueError('--check-initialization requires --method sugavanam_ertin alone')
    if args.se_stage1_only and methods != ['sugavanam_ertin']:
        raise ValueError('--se-stage1-only requires --method sugavanam_ertin alone')
    if args.config:
        if len(methods) != 1 or methods[0] in BUILTIN:
            raise ValueError('--config requires exactly one baseline; use --method-config for multiple baselines')
        options = {methods[0]: json.loads(args.config.read_text())}
    else:
        options = json.loads(args.method_config.read_text()) if args.method_config else {}
    if not isinstance(options, dict) or any(k not in methods or not isinstance(v, dict) for k,v in options.items()):
        raise ValueError('Method config must map selected method names to configuration objects')
    from rift import gotcha_baseline_planning as baseline_planning
    for method in methods:
        if method in baseline_planning.METHODS:
            options[method] = baseline_planning.validate_config(method, options.get(method, {}))
    dataset = GOTCHADataset(args.dataset_root, shard_root=args.shard_root,
                           passes=args.passes, polarizations=args.polarizations,
                           region=load_region(args.region,args.region_config), num_train=args.num_train,
                           num_tx=args.num_tx or 1, num_rx=args.num_rx or 1,
                           tx_indices=args.tx_indices, rx_indices=args.rx_indices,
                           pulses_per_sector=args.pulses_per_sector, frequency_stride=args.frequency_stride)
    if args.unit_split_stride:
        # Before any response read: the unit split re-labels rows and names itself in the contract.
        from rift_pvc.gotcha_unit_split import apply_unit_split
        apply_unit_split(dataset, stride=args.unit_split_stride, heldout_pass=args.unit_split_heldout_pass,
                         heldout_fraction=args.unit_split_heldout_fraction)
    from rift_pvc.gotcha_training import recipe_from_args
    from rift_pvc.gotcha_nufft import output_suffix
    from rift.gotcha_frequency_selection import bind_recipe
    plans = []
    # Full identity in the contract; a short digest keeps paths human-readable.
    root = args.output_root.absolute() / dataset.region.name / dataset.identity[:16]
    if args.pulses_per_sector:
        root = root/f'pulse_subset{args.pulses_per_sector}_all_roles_v2'
    if args.frequency_stride != 1:
        root = root/f'frequency_stride{args.frequency_stride}_all_roles_v2'
    for method in methods:
        spec = registry[method]
        if spec['builtin']:
            if method in options:
                raise ValueError('Use the documented CLI flags for built-in RIFT/MFBP recipes')
            config = bind_recipe(dataset, recipe_from_args(args,method))
            config = with_isolated_target(dataset, config)
        else:
            config = dict(options.get(method,{}))
        # The opt-in NUFFT/full-native control writes beside, never into, the default recipe's directory.
        method_root = root/(method + (output_suffix(config) if spec['builtin'] else ''))
        entry = dict(method=method, backend=spec, config=config,
                     output_dir=str(method_root), resume=str(args.resume) if args.resume else None)
        if args.se_stage1_only:
            entry['stage1_only'] = True
        if args.pulses_per_sector:
            entry['training_pulse_selection'] = dataset.training_pulse_selection
        if args.frequency_stride != 1:
            entry['training_frequency_selection'] = dataset.frequency_selection
        if method in baseline_planning.METHODS:
            entry.update(baseline_planning.make_plan(method, dataset, root/method, config))
        elif method == 'radar_fields':
            # PVC: the twin's defaults record model_backend upstream-tcnn-torchshim in the plan.
            from rift_pvc.radar_fields_gotcha import recipe_from_config
            entry['recipe'] = recipe_from_config(config, dataset.region.half_extent_m,
                                                 len(dataset.viewpoints('train')))
        elif method == 'spinr':
            # PVC: the twin's preflight records pulse_execution (the batched GEMM lane) in the plan.
            from rift_pvc.spinr_gotcha_training import preflight
            entry['native_plan'] = preflight(dataset, config)
        elif method == 'geraf':
            from rift_pvc.geraf_source import recipe_from_config   # the twin the run builds from
            entry['recipe'] = recipe_from_config(config, dataset.region.half_extent_m)
        plans.append(entry)
    return dataset, dict(dataset=dataset.summary(), plans=plans,
                         physical_antennas=dict(num_tx=1, num_rx=1, tx_indices=[0], rx_indices=[0],
                                                sampling=('same_pulse_and_native_bin_selection_train_validation_test' if args.frequency_stride != 1 else
                                                          'all_native_pulses_and_frequencies' if not args.pulses_per_sector
                                                          else 'same_fixed_pulse_cap_train_validation_test_all_native_frequencies')),
                         unavailable_methods={m:s['reason'] for m,s in registry.items() if s['status'] != 'available'},
                         selection_note=('All methods use the same pulse-cap/native-bin rules for train, validation and test. Normalization uses selected TRAIN only. Test payloads stay sealed during training/preparation.'
                                         if args.frequency_stride != 1 else
                                         'Counts refer to pass/sector viewpoints; all native pulses and frequencies are retained.'
                                         if not args.pulses_per_sector else
                                         'All methods use the same fixed native pulse cap in train/validation/test sectors. Normalization uses selected TRAIN pulses only; test payloads stay sealed during training.'))


def dispatch(dataset, plan, device):
    spec = plan['backend']
    module = importlib.import_module(spec['module'])
    if spec['builtin']:
        if plan['method'] == 'mfbp':
            return module.backproject(dataset,plan['config'],plan['output_dir'],device=device)
        return module.train(dataset,plan['method'],plan['config'],plan['output_dir'],device=device,resume=plan['resume'])
    run = getattr(module,spec.get('callable','run_gotcha'))
    kwargs = dict(dataset=dataset, output_dir=Path(plan['output_dir']), config=plan['config'],
                  device=device, resume=Path(plan['resume']) if plan['resume'] else None)
    if plan.get('stage1_only'):
        if plan['method'] != 'sugavanam_ertin':
            raise ValueError('Stage-1 boundary is only supported for Sugavanam–Ertin')
        from rift_pvc.sugavanam_ertin_paper_workflow import run_gotcha_stage1
        return run_gotcha_stage1(**kwargs)
    inspect.signature(run).bind(**kwargs)
    return run(**kwargs)


def main(argv=None):
    args = parse_args(argv)
    if args.list:
        registry = backend_registry()
        print(json.dumps(registry, indent=2))
        return
    dataset, plan = make_plan(args)
    if args.check_initialization:
        from rift.gotcha_baseline_planning import check_initialization
        entry = plan['plans'][0]
        entry['initialization_audit'] = check_initialization(dataset, entry)
    print(json.dumps(plan, indent=2), flush=True)
    if args.check_initialization:
        return 2 if entry['initialization_audit']['status'] == 'initialization_degenerate' else 0
    if args.dry_run:
        return plan
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Actual GOTCHA training/preparation requires an experiment-manager allocation; use --dry-run for local planning')
    # Preflight every destination before starting the first method.
    for entry in plan['plans']:
        path = Path(entry['output_dir'])
        if path.exists() and any(path.iterdir()) and not entry['resume']:
            raise ValueError(f'Existing output requires matching resume or a new --output-root: {path}')
        if entry['resume'] and not Path(entry['resume']).is_file():
            raise FileNotFoundError(entry['resume'])
    for entry in plan['plans']:
        result = dispatch(dataset,entry,args.device)
        print(json.dumps(dict(method=entry['method'], result=result), indent=2), flush=True)
        if isinstance(result,dict) and result.get('status') == 'interrupted':
            raise SystemExit(143)
    return plan


if __name__ == '__main__':
    result = main()
    raise SystemExit(result if isinstance(result, int) else 0)
