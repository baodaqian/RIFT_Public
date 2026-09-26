"""RadarSplat-owned collection routing; no data reads or process execution.

Keep model choices here so the shared dispatcher and other baseline owners do
not need to maintain RadarSplat recipes. Historical command budgets stay frozen.
"""
from pathlib import Path
import sys

from rift.rift_dataset import PROJECT_ROOT, object_paths, object_spec

DEFAULT_RECIPE = "budget48"
RECIPES = ("budget48", "upstream", "audit_v1", "legacy")


def add_arguments(parser):
    parser.add_argument("--radarsplat-recipe", choices=RECIPES, default=DEFAULT_RECIPE,
        help="RadarSplat only: budget48 uses source CUDA with 112000 Gaussians; upstream retains 20000")


def commands_for(name, *, dataset_root, output_root, resume=None, recipe=None, device=None, manifest_path=None):
    recipe = DEFAULT_RECIPE if recipe is None else recipe
    if recipe not in RECIPES:
        raise ValueError("Unknown RadarSplat recipe")
    if resume is not None and resume != "auto":
        raise ValueError("This method resumes its own output directory: use --resume auto")
    name = object_spec(name)["object_id"]
    npz, manifest = object_paths(Path(dataset_root), name)
    manifest = Path(manifest_path) if manifest_path is not None else manifest
    root = Path(output_root).absolute()/name/"radarsplat"
    if recipe == "budget48":
        root = root/recipe
    prepare = ["scripts/prepare_radarsplat_b7873200_targets.py", "--npz-path", str(npz),
               "--role-manifest", str(manifest), "--cache-root", str(root/"targets")]
    train = ["train_radarsplat.py", "--cache-root", str(root/"targets"),
             "--checkpoint-dir", str(root/"checkpoints"), "--fidelity-profile", recipe]
    if recipe not in ("upstream", "budget48"):
        train += ["--steps", "480000", "--validation-every", "600", "--checkpoint-every", "100",
                  "--init-num-gaussians", "2048", "--prune-every", "100"]
    if recipe in ("upstream", "budget48", "audit_v1"):
        prepare += ["--grid-policy", "scene_support", "--n-azimuth", "33", "--n-elevation", "33", "--n-range", "33"]
    train += ["--resume" if resume else "--no-resume"]
    if device:
        prepare += ["--device", device]
        train += ["--device", device]
    return [[sys.executable, str(PROJECT_ROOT/c[0]), *c[1:]] for c in (prepare, train)]
