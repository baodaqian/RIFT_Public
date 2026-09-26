#!/usr/bin/env python3
"""One object/method interface for the six-object homemade RIFT dataset.

Planning is read-only with --dry-run. Actual training and target preparation
must be executed by the experiment manager inside an allocated compute job.
No scenes, normalization statistics, target caches, or checkpoints are pooled.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

from rift.rift_dataset import (DEFAULT_ROOT, PROJECT_ROOT, catalog, object_paths,
                               object_spec, collection_contract, DEFAULT_NUM_TRAIN,
                               PARENT_NUM_TRAIN, training_count, role_manifest)

from rift.antenna_selection import add_arguments as antenna_arguments, from_args as antenna_from_args, acquisition_label

METHODS = ("rift", "rift_grid", "isotropic", "spinr", "radar_fields",
           "geraf", "radarsplat", "sugavanam_ertin", "sh_sas", "fsh", "mfbp")
ALIASES = {"adaptive": "rift", "adaptive_rift": "rift", "grid_sh": "rift_grid", "sp0": "isotropic", "rf": "radar_fields",
           "rs": "radarsplat", "se": "sugavanam_ertin", "sh-sas": "sh_sas"}
MERGED_BASELINES = ("radarsplat", "sugavanam_ertin")


def replace_option(argv: list[str], flag: str, value: object) -> None:
    if flag in argv:
        argv[argv.index(flag) + 1] = str(value)
    else:
        argv.extend((flag, str(value)))


def selected_output_root(output_root, num_train, antenna_selection=None):
    root = Path(output_root).absolute()
    root = root if num_train == PARENT_NUM_TRAIN else root / f"train{num_train}"
    return root / acquisition_label(antenna_selection) if antenna_selection else root


def selected_manifest(dataset_root, output_root, name, num_train, antenna_selection=None):
    if num_train == PARENT_NUM_TRAIN and antenna_selection is None:
        return object_paths(dataset_root, name)[1]
    return selected_output_root(output_root, num_train, antenna_selection) / object_spec(name)["object_id"] / "role_manifest.json"


def write_selected_manifest(path, manifest):
    """Publish complete role metadata atomically without replacing another run."""
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError(f"Existing subset manifest changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.role_manifest.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(manifest, stream, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if json.loads(path.read_text()) != manifest:
                raise ValueError(f"Existing subset manifest changed: {path}")
    finally:
        Path(temporary).unlink()


def commands_for(name: str, method: str, *, dataset_root: Path, output_root: Path,
                 resume: str | None = None, host_rss_limit_gib: float = 48.0,
                 spinr_recipe: str = "budget48-direct", radarsplat_recipe: str | None = None,
                 geraf_implementation: str = "source_v1", geraf_source_config: Path | None = None,
                 radar_fields_recipe: str = "source-adapted-v3",
                 se_recipe: str | None = None, se_config: Path | None = None,
                 se_stage1_only: bool = False,
                 device: str | None = None, check_initialization: bool = False,
                 num_train: int = PARENT_NUM_TRAIN, antenna_selection=None) -> list[list[str]]:
    """Build ordinary maintained-trainer commands; no subprocess/data access here."""
    name = object_spec(name)["object_id"]
    num_train = training_count(num_train)
    manifest = selected_manifest(dataset_root, output_root, name, num_train, antenna_selection)
    output_root = selected_output_root(output_root, num_train, antenna_selection)
    method = ALIASES.get(method, method)
    if method == "geraf" and geraf_source_config is not None and geraf_implementation != "source_v1":
        raise ValueError("--geraf-source-config applies only to source_v1")
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}")
    # Baseline owners maintain their routing separately. Keep this socket free
    # of model budgets/defaults so concurrent work cannot change another recipe.
    if check_initialization and method != "sugavanam_ertin":
        raise ValueError("Initialization probe requires Sugavanam–Ertin alone")
    if method == "radarsplat":
        from rift import radarsplat_collection as owner
        return owner.commands_for(name, dataset_root=dataset_root, output_root=output_root,
                                  resume=resume, recipe=radarsplat_recipe, device=device, manifest_path=manifest)
    if method == "sugavanam_ertin":
        from rift import sugavanam_ertin_collection as owner
        return owner.commands_for(name, dataset_root=dataset_root, output_root=output_root,
                                  resume=resume, recipe=se_recipe, config=se_config,
                                  device=device, check_initialization=check_initialization, manifest_path=manifest,
                                  stage1_only=se_stage1_only)
    npz, _ = object_paths(dataset_root, name)
    object_root = output_root.absolute() / name
    run_root = object_root / method
    data = ["--npz-path", str(npz)]
    counts = ["--num-train", str(num_train), "--num-val", "1000", "--num-test", "1000", "--seed", "42"]
    output = ["--checkpoint-root", str(object_root), "--checkpoint-name", method]
    commands = []
    if method in ("rift_grid", "isotropic"):
        argv = ["train.py", "--data-format", "npz", *data, "--npz-sealed-protocol",
                "--npz-role-manifest", str(manifest), *counts, *output,
                "--scene-repr", "grid_sh" if method == "rift_grid" else "grid",
                "--forward-operator", "range", "--compute-dtype", "float64",
                "--range-model", "sum2", "--phase-sign", "-1.0", "--num-freq-wanted", "600",
                "--num-tx", "16", "--num-rx", "16", "--extent", "0.15", "--granularity", "48",
                "--epochs", "150", "--bp-init", "100", "--lr", "0.003", "--adam-eps", "1e-20",
                "--l1-weight", "3e-7", "--sh-max-degree", "3", "--sh-init-degree", "3",
                "--prune-every", "0", "--grow-every", "0", "--checkpoint-metric", "val"]
    elif method == "rift":
        from rift.b7873200_adaptive_fullscale import fullscale_train_argv
        argv = fullscale_train_argv(npz_path=npz, manifest_path=manifest, checkpoint_root=object_root)
        replace_option(argv, "--checkpoint-name", method)
        replace_option(argv, "--execution-contract-label", "rift_dataset_adaptive_v1")
        if num_train != PARENT_NUM_TRAIN:
            from rift.collection_adaptive import execution_label
            replace_option(argv, "--num-train", num_train)
            replace_option(argv, "--execution-contract-label", execution_label(num_train))
        argv.insert(0, "train.py")
    elif method == "spinr":
        if num_train < 32 or num_train % 4:
            raise ValueError("SpINR requires at least 32 training views in complete four-view batches")
        if spinr_recipe not in ("budget48-direct", "budget48-direct-1500", "paper-v1-direct", "paper-v1", "legacy-midpoint"):
            raise ValueError("Unknown SpINR recipe")
        if spinr_recipe in ("budget48-direct", "budget48-direct-1500"):
            name = "budget48-direct-150" if spinr_recipe == "budget48-direct" else "budget48-direct"
            output = ["--checkpoint-root", str(run_root), "--checkpoint-name", name]
        argv = ["train_spinr_style.py", *data, "--npz-role-manifest", str(manifest), *output,
                "--host-rss-limit-gib", str(host_rss_limit_gib), "--recipe", spinr_recipe]
    elif method == "radar_fields":
        if radar_fields_recipe not in ("source-adapted-v3", "audited-v2", "legacy-v1"):
            raise ValueError("Unknown Radar Fields recipe")
        argv = ["train_radar_fields.py", "--recipe", radar_fields_recipe, *data, *counts, *output, "--sealed-protocol",
                "--sealed-split-manifest", str(manifest), "--extent", "0.15", "--granularity", "48",
                "--steps", "8000", "--view-batch", "1", "--train-pairs", "0", "--val-pairs", "0",
                "--eval-every", "2000", "--checkpoint-every", "500"]
        if radar_fields_recipe in ("audited-v2", "source-adapted-v3"):
            argv += ["--device", "cuda"]
        if radar_fields_recipe == "source-adapted-v3":
            if num_train % 10:
                raise ValueError("Radar Fields source recipe requires complete ten-view batches")
            batches = num_train // 10
            for flag, value in (("--steps", batches * math.ceil(800 / batches)), ("--view-batch", 10), ("--train-pairs", 100),
                                ("--seed", 0), ("--eval-every", batches), ("--checkpoint-every", batches)):
                replace_option(argv, flag, str(value))
            argv += ["--ray-samples", "10"]
    elif method == "sh_sas":
        argv = ["train_sh_sas.py", *data, *counts, *output, "--npz-role-manifest", str(manifest),
                "--extent", "0.15", "--granularity", "48", "--num-freq-wanted", "600",
                "--phase-sign", "-1.0", "--compute-dtype", "fp64"]
    elif method == "geraf" and geraf_implementation == "source_v1":
        run_root = run_root / "source_v1"
        argv = ["train_geraf.py", "--implementation", "source_v1", *data,
                "--role-manifest", str(manifest), "--cache-root", str(run_root / "targets"),
                "--checkpoint-dir", str(run_root / "checkpoints"),
                "--resume" if resume else "--no-resume"]
        if geraf_source_config is not None:
            argv += ["--source-config", str(geraf_source_config)]
    elif method == "geraf":
        cache = run_root / "targets"
        commands.append([f"scripts/prepare_{method}_b7873200_targets.py", *data,
                         "--role-manifest", str(manifest), "--cache-root", str(cache)])
        argv = [f"train_{method}.py", "--cache-root", str(cache),
                "--checkpoint-dir", str(run_root / "checkpoints")]
        if geraf_implementation not in {"hardened_v1", "legacy"}:
            raise ValueError("Unknown GeRaF implementation")
        argv += [*data, "--role-manifest", str(manifest), "--implementation", geraf_implementation]
        argv += ["--resume" if resume else "--no-resume"]
    elif method in ("fsh", "mfbp"):
        argv = ["scripts/eval_rift_dataset_model_free.py", *data, "--role-manifest", str(manifest),
                "--method", method, "--output-dir", str(run_root)]
    if resume:
        if method == "geraf":
            if resume != "auto":
                argv += ["--resume-path", resume]
        elif method in ("fsh", "mfbp"):
            if resume != "auto":
                raise ValueError("This method resumes its own output directory: use --resume auto")
            if method in ("fsh", "mfbp"):
                argv += ["--resume"]
        else:
            if resume == "auto":
                raise ValueError("This method requires an explicit --resume checkpoint path")
            argv += ["--resume", str(Path(resume).absolute())]
    if method in ("rift", "rift_grid", "isotropic") and antenna_selection:
        replace_option(argv, "--num-tx", antenna_selection["num_tx"])
        replace_option(argv, "--num-rx", antenna_selection["num_rx"])
    commands.append(argv)
    return [[sys.executable, str(PROJECT_ROOT / command[0]), *command[1:]] for command in commands]


def preflight_object(root: Path, name: str, num_train=PARENT_NUM_TRAIN, antenna_selection=None) -> dict:
    from rift.rift_dataset import load_object
    options = {} if antenna_selection is None else {k: antenna_selection[k] for k in
        ("num_tx", "num_rx", "tx_indices", "rx_indices")}
    _, contract = load_object(name, root, num_train=num_train, **options)
    identity = collection_contract(contract)
    if identity is None:
        raise ValueError("Prepare the named RIFT dataset first; a legacy manifest is not interchangeable")
    return identity


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    antenna_arguments(parser)
    parser.add_argument("--object", nargs="+", default=["all"], help="Object IDs or a320/x59/racecar; all selects six")
    parser.add_argument("--method", nargs="+", default=["rift"], help="Method names; all selects nine radar trainers plus FSH/MFBP")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--num-train", type=int, default=DEFAULT_NUM_TRAIN,
                        help="Nested training views per object (1..3200; default 2400); validation/test stay 1000/1000")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "training_checkpoints" / "RIFT_dataset")
    parser.add_argument("--dry-run", action="store_true", help="Validate metadata/splits and print commands, without writes or training")
    parser.add_argument("--list", action="store_true", help="List objects/methods without requiring data")
    parser.add_argument("--resume", help="One object/method only: explicit checkpoint, or auto for GeRaF/RadarSplat")
    parser.add_argument("--se-config", "--config", dest="se_config", type=Path,
                        help="SE paper-v1 recipe JSON; --config preserves the old SE frontend flag")
    parser.add_argument("--device",
                        help="Explicit execution device for RadarSplat/SE selections")
    parser.add_argument("--check-initialization", action="store_true", help="SE only: bounded CPU probe, no fitting")
    parser.add_argument("--host-rss-limit-gib", type=float, default=48.0, help="SpINR host-memory limit")
    parser.add_argument("--spinr-recipe", choices=("budget48-direct", "budget48-direct-1500", "paper-v1-direct", "paper-v1", "legacy-midpoint"), default="budget48-direct",
                        help="SpINR only: G48 direct renderer/150 epochs, or an explicit historical checkpoint recipe")
    from rift import radarsplat_collection, sugavanam_ertin_collection
    radarsplat_collection.add_arguments(parser)
    parser.add_argument("--geraf-implementation", choices=("source_v1", "hardened_v1", "legacy"), default="source_v1",
                        help="GeRaF only: released source with v1 settings, or an explicit historical adaptation")
    parser.add_argument("--geraf-source-config", type=Path,
                        help="GeRaF source_v1 only: JSON mapping of explicit recipe overrides")
    parser.add_argument("--radar-fields-recipe", choices=("source-adapted-v3", "audited-v2", "legacy-v1"), default="source-adapted-v3",
                        help="Radar Fields only: source-adapted-v3 comparison profile, or explicit historical recipes")
    sugavanam_ertin_collection.add_arguments(parser)
    return parser.parse_args(argv)


def make_plan(args):
    num_train = training_count(args.num_train)
    antenna_selection = antenna_from_args(args, default=1)
    names = ([s["object_id"] for s in catalog()["objects"]] if args.object == ["all"]
             else [object_spec(name)["object_id"] for name in args.object])
    methods = list(METHODS) if args.method == ["all"] else [ALIASES.get(m, m) for m in args.method]
    if len(set(names)) != len(names) or len(set(methods)) != len(methods):
        raise ValueError("Do not select the same object/method twice")
    if not methods or any(method not in METHODS for method in methods):
        raise ValueError("Select known method names")
    if args.resume and len(names) * len(methods) != 1:
        raise ValueError("Resume requires exactly one object and one method")
    if args.check_initialization and methods != ["sugavanam_ertin"]:
        raise ValueError("--check-initialization requires --method sugavanam_ertin alone")
    if args.device and any(m not in MERGED_BASELINES for m in methods):
        raise ValueError("--device currently applies to RadarSplat/SE selections only")
    if args.se_config and "sugavanam_ertin" not in methods:
        raise ValueError("--se-config/--config requires selecting Sugavanam–Ertin")
    if args.se_stage1_only and methods != ["sugavanam_ertin"]:
        raise ValueError("--se-stage1-only requires --method sugavanam_ertin alone")
    if "sugavanam_ertin" in methods:
        if args.se_recipe != "paper-v1" and (args.se_config or args.check_initialization):
            raise ValueError("Configuration/probe options belong to paper-v1, not the frozen legacy recipe")
        if args.se_recipe == "paper-v1":
            from rift.sugavanam_ertin_paper_workflow import make_recipe
            make_recipe("rift_collection", json.loads(args.se_config.read_text()) if args.se_config else {})
    plans = []
    for name in names:
        identity = preflight_object(args.dataset_root, name, num_train, antenna_selection)
        manifest = selected_manifest(args.dataset_root, args.output_root, name, num_train, antenna_selection)
        if (num_train != PARENT_NUM_TRAIN or antenna_selection) and manifest.exists():
            if json.loads(manifest.read_text()) != role_manifest(name, num_train, antenna_selection):
                raise ValueError(f"Existing subset manifest changed: {manifest}")
        for method in methods:
            commands = commands_for(name, method, dataset_root=args.dataset_root,
                                    output_root=args.output_root, resume=args.resume,
                                    host_rss_limit_gib=args.host_rss_limit_gib,
                                    spinr_recipe=args.spinr_recipe,
                                    radarsplat_recipe=args.radarsplat_recipe,
                                    geraf_implementation=args.geraf_implementation,
                                    geraf_source_config=args.geraf_source_config,
                                    radar_fields_recipe=args.radar_fields_recipe,
                                    se_recipe=args.se_recipe, se_config=args.se_config,
                                    se_stage1_only=args.se_stage1_only,
                                    device=args.device, check_initialization=args.check_initialization,
                                    num_train=num_train, antenna_selection=antenna_selection)
            run_root = selected_output_root(args.output_root, num_train, antenna_selection) / name / method
            if method == "geraf" and args.geraf_implementation == "source_v1":
                run_root = run_root / "source_v1"
            entry = dict(object=name, method=method, dataset_identity=identity,
                         role_manifest_path=str(selected_manifest(args.dataset_root, args.output_root, name, num_train, antenna_selection)),
                         output_dir=str(run_root), commands=commands)
            if method == 'geraf' and args.geraf_implementation == 'source_v1':
                from types import SimpleNamespace
                from rift.geraf_source import recipe_for_data
                config = json.loads(args.geraf_source_config.read_text()) if args.geraf_source_config else {}
                entry['recipe'] = recipe_for_data(config, SimpleNamespace(
                    contract={'experiment_contract': identity}, extent=float(catalog()['scene_extent_m'])))
            if method in MERGED_BASELINES:
                entry['recipe'] = args.radarsplat_recipe if method == 'radarsplat' else args.se_recipe
            plans.append(entry)
    return dict(entrypoint="train_rift_dataset.py", num_train=num_train, antenna_selection=antenna_selection, plans=plans, response_payload_read=False)


def main(argv=None):
    args = parse_args(argv)
    if args.list:
        print(json.dumps({"dataset": "RIFT dataset", "objects": [s["object_id"] for s in catalog()["objects"]],
                          "methods": METHODS}, indent=2))
        return 0
    plan = make_plan(args)
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return 0
    if not args.check_initialization and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Training/preparation requires an experiment-manager allocation; use --dry-run")
    # Check every object/method before the first subprocess can create outputs.
    if not args.check_initialization and not args.resume:
        for entry in plan['plans']:
            path = Path(entry['output_dir'])
            occupied = (path.exists() and any(p.name != "targets" or entry['method'] in MERGED_BASELINES
                                              for p in path.iterdir()))
            if occupied:
                raise ValueError(f"Existing output requires explicit --resume or a new output root (--output-root): {path}")
    # Preserve the RadarSplat frontend's dependency gate before target conversion.
    if any(e['method'] == 'radarsplat' and e['recipe'] in ('upstream', 'budget48') for e in plan['plans']):
        from rift.radarsplat_release import load_cuda_reference
        load_cuda_reference(device=args.device or 'cuda')
    # Derived manifests are local run inputs. Parent data/splits are never rewritten.
    if args.num_train != PARENT_NUM_TRAIN or plan["antenna_selection"]:
        for entry in plan['plans']:
            write_selected_manifest(entry['role_manifest_path'], role_manifest(entry['object'], args.num_train, plan['antenna_selection']))
    for entry in plan['plans']:
        for command in entry['commands']:
            print(shlex.join(command), flush=True)
            result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
            if result.returncode:
                return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
