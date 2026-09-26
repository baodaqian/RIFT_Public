"""Sugavanam--Ertin-owned collection routing; no data reads or execution.

Extracted without changing the maintained SE model, CLI or checkpoint recipe.
Subsequent SE changes belong here, independently of other baseline owners.
"""
from pathlib import Path
import sys

from rift.rift_dataset import PROJECT_ROOT, object_paths, object_spec

DEFAULT_RECIPE = "paper-v1"
RECIPES = ("paper-v1", "legacy-full")


def add_arguments(parser):
    parser.add_argument("--se-stage1-only", action="store_true",
        help="SE paper-v1 only: stop after sparse Stage 1, before SDF fitting")
    parser.add_argument("--se-recipe", choices=RECIPES, default=DEFAULT_RECIPE,
        help="SE only: sub-aperture/SDF recipe, or historical isotropic/stabilized recipe")


def commands_for(name, *, dataset_root, output_root, resume=None, recipe=None,
                 config=None, device=None, check_initialization=False, manifest_path=None,
                 stage1_only=False):
    recipe = DEFAULT_RECIPE if recipe is None else recipe
    if recipe not in RECIPES:
        raise ValueError("Unknown Sugavanam--Ertin recipe")
    if stage1_only and recipe != "paper-v1":
        raise ValueError("Stage-1-only execution requires paper-v1")
    if recipe != "paper-v1" and (config or check_initialization):
        raise ValueError("Configuration/probe options belong to paper-v1, not the frozen legacy recipe")
    if resume == "auto":
        raise ValueError("This method requires an explicit --resume checkpoint path")
    name = object_spec(name)["object_id"]
    npz, manifest = object_paths(Path(dataset_root), name)
    manifest = Path(manifest_path) if manifest_path is not None else manifest
    root = Path(output_root).absolute()/name/"sugavanam_ertin"
    command = [sys.executable, str(PROJECT_ROOT/"train_sugavanam_ertin.py"),
               "--recipe", recipe, "--npz-path", str(npz), "--parent-role-manifest", str(manifest),
               "--checkpoint-root", str(root)]
    if resume:
        command += ["--resume", str(Path(resume).absolute())]
    if device:
        command += ["--device", device]
    if config:
        command += ["--config", str(Path(config).absolute())]
    if check_initialization:
        command += ["--check-initialization"]
    if stage1_only:
        command += ["--stage1-only"]
    return [command]
