"""Unmodified released RF learning core for source-format measurement batches.

This deliberately does not invent a Navtech-image adapter for coherent MIMO.
The existing audited-v2 adapter remains a separate engineering recipe. Native
batch inputs must contain the real acquisition calibration and preprocessing.
"""

from __future__ import annotations

import ast
import copy
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .radar_fields_upstream import REFERENCE_ROOT, original_module, verify_sources


def released_arguments():
    """Resolve the author's actual parser/config, including enabled priors."""
    parser = original_module("parse").get_arg_parser()
    return parser.parse_args(["--config", str(REFERENCE_ROOT / "configs/radarfields.ini")])


@lru_cache(maxsize=1)
def released_optimizer_factories():
    """Evaluate the two original lambda expressions without loading a dataset.

    The AST expressions are taken verbatim from hash-verified main.py. This
    avoids reimplementing optimizer groups, optimizer choice, or the LR clock.
    No other statement from main.train is executed.
    """
    verify_sources()
    path = REFERENCE_ROOT / "main.py"
    function = next(node for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "train")
    expressions = {}
    for node in function.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("optimizer", "scheduler"):
                if not isinstance(node.value, ast.Lambda):
                    raise RuntimeError("pinned RF optimizer factory is no longer a lambda")
                expressions[name] = compile(ast.Expression(node.value), str(path), "eval")
    if set(expressions) != {"optimizer", "scheduler"}:
        raise RuntimeError("missing original RF optimizer/scheduler expressions")
    return expressions


def optimizer_factories(args):
    namespace = {"torch": torch, "args": args}
    expressions = released_optimizer_factories()
    return tuple(eval(expressions[name], namespace) for name in ("optimizer", "scheduler"))


def released_epoch_count(args, frame_count):
    """Original ceil(iters / len(loader)), including its final full epoch."""
    batches = int(np.ceil(frame_count / args.bs))
    if batches < 1:
        raise ValueError("a source run requires training frames")
    return np.ceil(args.iters / batches).astype(np.int32)


def make_released_trainer(*, workspace, all_poses, heldout_indices, device="cuda",
                          model=None):
    """Build the actual original Trainer; writes only to the fresh workspace.

    With model=None this instantiates the original TCNN RadarField directly,
    with no cast, direction renormalization, boundary clamp, or BN wrapper.
    An explicitly supplied model is for synthetic source-parity tests, and is
    never a fallback selected by the production caller. No dataset is opened.
    The caller must supply a loader with the original RadarDataset batch schema.
    """
    args = released_arguments()
    args.workspace = str(Path(workspace).absolute())
    output = Path(args.workspace)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("source reference requires a fresh workspace; do not mix adapter checkpoints")
    args.device = torch.device(device)
    args.all_poses = all_poses.to(args.device)
    args.test_indices = list(heldout_indices)
    original_module("utils.train").seed_everything(args.seed)
    if model is None:
        if args.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("the released TCNN model requires an allocated CUDA device")
        model = original_module("radarfields.nn.models").RadarField(
            **copy.deepcopy(args.model_settings), use_tcnn=args.tcnn)
    optimizer, scheduler = optimizer_factories(args)
    trainer_type = original_module("radarfields.train").Trainer
    criteria = {"fft": torch.nn.L1Loss(), "occ": torch.nn.KLDivLoss(reduction="batchmean")}
    return trainer_type(args, model, split="train", criterion=criteria,
                        optimizer=optimizer, lr_scheduler=scheduler, device=args.device)


def source_fidelity_inventory():
    """Read-only facts from the release; no locally selected training settings."""
    args = released_arguments()
    fields = ("seed", "iters", "bs", "num_rays_radar", "num_fov_samples",
              "num_range_samples", "min_range_bin", "max_range_bin", "lr",
              "mask", "integrate_rays", "approximate_fft", "train_thresholded",
              "learned_norm", "initial_offset", "initial_scaler", "reg_occ",
              "weight_fft", "weight_occ", "bimodal", "weight_bimodal", "ground_occ",
              "weight_ground_occ", "penalize_above", "weight_above", "refine_poses",
              "pose_mode", "pose_lr", "schedule_pose", "noise_floor")
    return {
        "source_commit": "ee76d76570f58b3d8539eafd7df0c188b58af333",
        "configuration": "configs/radarfields.ini plus parse.py defaults",
        "settings": {name: getattr(args, name) for name in fields},
        "model_settings": copy.deepcopy(args.model_settings),
        "intrinsics_radar": copy.deepcopy(args.intrinsics_radar),
        "mask_clock": "original Trainer.train: epoch, not optimizer step",
        "optimizer_clock": "iters is the LR clock; original training completes ceil(iters/len(loader)) full epochs",
        "frame_sampler": "torch SubsetRandomSampler, without replacement per epoch; no drop_last",
        "azimuth_sampler": "original sorted torch.randint: sampling WITH replacement",
        "ray_sampler": "original pitch/yaw uniform samples, one central ray, pitch sorting",
        "loss": "original Trainer.compute_loss, including ground/above terms and nan_to_num",
        "pose": "original PoseOptimizer and original held-out pose interpolation",
        "rift_adapter_source_equivalent": False,
    }
