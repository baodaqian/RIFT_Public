#!/usr/bin/env python
"""RIFT training entrypoint: adaptive point-SH by default.

Select --architecture adaptive, grid_sh, grid, point_sh, or mlp. The adaptive
preset uses the reviewed joint spatial/angular capacity-v2 recipe. Explicit
legacy --scene-repr commands keep their original defaults, so existing fixed
recipes do not silently become adaptive. Use --object to bind a RIFT dataset
scene and its sealed roles. Adaptive technical reporting/recovery is available
through --workflow adaptive-fullscale.

Maintained RIFT trainer. Replaces the retired hand-copied per-job trainers;
their loss weights, cosine-restart periods, data paths, and checkpoint names
are CLI options here. See docs/CLEANUP_20260919.md for their recovery archive.

Usage:
    python train.py --object a320 --checkpoint-name rift
    python train.py --object b787 --architecture grid_sh --checkpoint-name grid_control

2026-07-03 objective/optimizer rework (see EXPERIMENT_MANAGER_HANDOFF.md,
"Pilot non-convergence diagnosis"): the historical magnitude+wrapped-phase
MSE objective sits at its random-phase noise floor (w2 * 2pi^2/3) and cannot
be descended -- all pre-2026-07-03 runs were flat because of it. Current
defaults: complex-residual loss (--loss complex), one optimizer step per
viewpoint (--step-every 1), learnable global calibration gain, no weight
decay, no grad clipping. The exact legacy behavior remains reproducible via:
    --loss magphase --step-every 0 --clip-grad-norm 1.0 \\
    --weight-decay 1e-3 --no-learn-gain

See slurm/train.sbatch for the corresponding Slurm submission template.
"""
import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from collections.abc import Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from rift.config import arr_dist, extent, fp_granularity, get_freqs, num_rx, num_tx, pos_encoding_degree, spacing, cc
from rift.calibration import GlobalComplexGain
from rift.checkpointing import generate_loss_path, save_checkpoint, save_losses
from rift.dataset import CSVSimulationDataset, list_and_select_files
from rift.distributed import (
    all_reduce_int,
    all_reduce_max,
    all_reduce_sum,
    all_reduce_sum_grad,
    barrier,
    get_rank,
    get_world_size,
    init_distributed,
    is_dist,
    rank0_print,
    shutdown_distributed,
)
from rift.encoding import generate_dynamic_grid, positional_encoding, prepare_model_input
from rift.forward_operator import (
    adjoint_operator_lessparallel,
    forward_operator_lessparallel,
    get_array_pos,
    get_kvector,
)
from rift.model import MLP
from rift.npz_dataset import (
    PecSphereNPZDataset,
    build_npz_dataloaders,
    load_npz_arrays,
    npz_response_num_views,
    restrict_npz_response_views,
)
from rift.occlusion import OcclusionScale, array_phase_centre, view_transmittance
from rift.range_operator import range_adjoint_operator, range_forward_operator
from rift.sharded_scene import ShardedSHVoxelGridScene
from rift.sparse_scene import AdaptivePointSHScene, SHVoxelGridScene, VoxelGridScene
from rift.spherical_harmonics import num_sh_basis, physical_max_degree

# Multi-GPU runs execute this whole module on every rank; unguarded prints
# would emit one interleaved copy per GPU. rank0_print no-ops off rank 0 and
# is a plain print in single-process runs, so the log is unchanged.
print = rank0_print

# Anything that changes the LOSS must be drawn identically on every rank.
# The per-device CUDA RNG cannot be trusted for that once shard shapes
# differ (each rank consumes a different number of values at scene init), so
# the random frequency subset comes from this shared CPU generator instead.
# For the current npz recipes (--num-freq-wanted 600 of 600 bins) the
# selection is the full sorted set either way, so single-GPU runs reproduce
# exactly; with a strict subset the draw differs from pre-2026-07 runs while
# remaining an unbiased uniform subset.
_FREQ_RNG = torch.Generator(device="cpu")


def _sh_max_degree_arg(value):
    """argparse type for --sh-max-degree: an int 0-10, or the literal 'auto'
    (resolved in main() once the data's f_max and the scene pitch are known)."""
    if isinstance(value, str) and value.strip().lower() == "auto":
        return "auto"
    try:
        degree = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"--sh-max-degree must be an int 0-10 or 'auto', got {value!r}")
    if not 0 <= degree <= 10:
        raise argparse.ArgumentTypeError(f"--sh-max-degree must be 0-10 or 'auto', got {degree}")
    return degree


def _nonnegative_float_arg(value):
    """Argparse type for loss weights that must be finite and non-negative."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"expected a non-negative finite float, got {value!r}")
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative finite float, got {value!r}")
    return number


def resolve_sh_max_degree(requested, extent_m, granularity, f_max_hz, safety=1.0):
    """Turn --sh-max-degree ('auto' or an int) into a concrete degree, and
    return (degree, message) so callers can log the reasoning. Warns when an
    explicitly-requested degree exceeds what one voxel can physically carry --
    the excess coefficients only fit training views (see the 2026-07-30
    handoff section for the measured train/val gap this produces)."""
    pitch = 2.0 * extent_m / granularity
    phys = physical_max_degree(pitch, f_max_hz, c=cc, safety=safety)
    if requested == "auto":
        return phys, (f"--sh-max-degree auto -> {phys} (pitch {1e3 * pitch:.2f} mm, "
                      f"f_max {f_max_hz / 1e9:.2f} GHz, safety {safety:g}); "
                      f"{num_sh_basis(phys)} coefficients per entry")
    msg = (f"--sh-max-degree {requested} ({num_sh_basis(requested)} coefficients per entry); "
           f"physics allows {phys} at pitch {1e3 * pitch:.2f} mm")
    if requested > phys:
        msg += (f" -- WARNING: over the physical angular bandwidth of one voxel by "
                f"{requested - phys} degree(s) ({num_sh_basis(requested) - num_sh_basis(phys)} "
                f"excess coefficients per entry). These cannot describe a physical scattering "
                f"pattern and generalize poorly; consider 'auto'.")
    return requested, msg


@torch.no_grad()
def normalize_scene_scale(model, gain, target_rms=1.0):
    """Fix the (gain, scene) scale gauge at init.

    The objective is EXACTLY invariant under (g -> g/a, w -> a*w): the data
    term because the operator is linear in w, and the regularizers because
    they already carry an explicit |g| / |g|^2 factor (see
    regularization_loss). That flat direction is unconstrained, and it shows:
    across the checkpoints on disk |g| spans 2e-4 to 1e5, and over the B787
    density sweep |g| fell 2x while rms|w| rose 5.9x -- the two traded along
    the gauge instead of converging.

    The cost is that |w| is incomparable across runs and across granularity,
    because the natural per-entry magnitude falls as the same physical target
    is split over more scatterers (the forward operator SUMS them).

    Fix it once, after init, by rescaling to rms|w| = target_rms over the
    ACTIVE entries and folding the reciprocal into the gain. Exactly
    loss-preserving. Done before the optimizer is built, so there is no Adam
    moment state to rescale; skipped on --resume, where the checkpoint's
    gauge must be preserved.

    CRITICAL -- the caller MUST scale the scene's learning rate by the same
    factor `a` that this returns. The objective is gauge-invariant but the
    OPTIMIZER is not: AdamW's step is ~lr in ABSOLUTE units, so the relative
    step on a parameter of size |w| is lr/|w|, and rescaling w -> a*w divides
    the scene's effective learning rate by a. An earlier version of this
    docstring claimed the opposite ("Adam normalizes per parameter, so the
    step stops depending on the gauge") -- that is backwards. Adam is
    invariant to rescaling the GRADIENT, not the PARAMETER.

    Round 3 (2026-08-01) paid for that error: a = 318 on B787 g48, so the
    scene's effective lr fell 318x, rms|w| moved 0.24% in 150 epochs, and with
    the scene frozen the only way left to cut the loss was to shrink the gain
    -- all five runs walked to the predict-zero floor (val 100.1% against a
    38.5% baseline). See the 2026-08-02 handoff section.

    Deliberately NOT a fixed pitch^2 / pitch^3 normalization: whether the
    right power is area or volume depends on whether the target is a surface
    or a volume, and the scene has no way to know. Normalizing empirically to
    the scene's own rms sidesteps that entirely.
    """
    if gain is None or not hasattr(model, "w_re"):
        return None
    mask = getattr(model, "active_mask", None)
    sq = model.w_re ** 2 + model.w_im ** 2
    if sq.dim() > (mask.dim() if mask is not None else 0):
        sq = sq.sum(dim=-1)
    if mask is not None:
        sq = sq[mask]
    # The rms must be over the GLOBAL active set: ShardedSHVoxelGridScene holds
    # one voxel shard per rank, and a per-rank rms would give every rank a
    # DIFFERENT rescale factor while the gain (replicated) got only one of them
    # -- silently corrupting the scene. Reduce the sum and the count separately;
    # both are exact and the whole operation stays loss-preserving.
    # Accumulate in fp64: the coefficients are stored fp32, and an fp32 sum
    # reassociates differently on 1 rank vs N (the documented ~8e-8 residual).
    local_sq = float(sq.double().sum()) if sq.numel() else 0.0
    n_active = all_reduce_int(int(sq.numel()), device=model.w_re.device)
    total_sq = float(all_reduce_sum(torch.tensor(local_sq, device=model.w_re.device, dtype=torch.float64)))
    if n_active == 0:
        return None
    rms = math.sqrt(total_sq / n_active)
    if not (rms > 0) or not math.isfinite(rms):
        rank0_print(f"Scene-scale gauge: rms|w| = {rms:.3e}, not normalizable; leaving as-is.")
        return None
    a = target_rms / rms
    model.w_re *= a
    model.w_im *= a
    gain.log_mag -= math.log(a)
    rank0_print(f"Scene-scale gauge: rms|w| {rms:.4e} -> {target_rms:.1f} "
                f"(x{a:.4e}), |g| folded to {float(torch.exp(gain.log_mag)):.4e}. "
                f"Exactly loss-preserving; makes |w| comparable across granularity.")
    return a


def select_freq_indices(n_freqs, num_wanted, device):
    """Sorted random subset of frequency bins, identical across ranks."""
    k = min(num_wanted, n_freqs)
    perm = torch.randperm(n_freqs, generator=_FREQ_RNG)
    return torch.sort(perm[:k])[0].to(device)


def set_seed(seed_value=42):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    _FREQ_RNG.manual_seed(seed_value)


def _serialize_numpy_rng_state():
    """Encode NumPy's legacy global RNG without pickled NumPy objects.

    PyTorch 2.6's restricted checkpoint loader accepts tensors and primitive
    containers but intentionally rejects a pickled ``numpy.ndarray``.  The
    old ``np.random.get_state()`` tuple contains exactly such an array.  Keep
    the same RandomState state bit-for-bit, but express its key vector as a
    CPU tensor so ordinary evaluation/checkpoint readers remain safe.
    """
    algorithm, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    keys = np.asarray(keys, dtype=np.uint32)
    return {
        "format": "numpy_randomstate_v1",
        "algorithm": str(algorithm),
        "keys": torch.from_numpy(keys.copy()),
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def _decode_numpy_rng_state(payload):
    """Recover a NumPy RNG tuple, moving map-located tensor state to CPU."""
    if isinstance(payload, dict) and payload.get("format") == "numpy_randomstate_v1":
        required = {"algorithm", "keys", "position", "has_gauss", "cached_gaussian"}
        missing = required - set(payload)
        if missing:
            raise ValueError(f"NumPy RNG payload is missing {sorted(missing)}")
        keys = payload["keys"]
        if not torch.is_tensor(keys) or keys.ndim != 1:
            raise ValueError("NumPy RNG key state must be a one-dimensional tensor")
        # map_location=device may put all saved tensors on CUDA.  NumPy is
        # CPU-only, and this explicit transfer is part of exact CUDA resume.
        key_array = keys.detach().to(device="cpu", dtype=torch.int64).numpy().astype(
            np.uint32, copy=True)
        return (
            str(payload["algorithm"]),
            key_array,
            int(payload["position"]),
            int(payload["has_gauss"]),
            float(payload["cached_gaussian"]),
        )
    # Direct callers can still supply the historical tuple.  Checkpoint files
    # are loaded with ``weights_only=True`` below, so an unsafe pickled tuple
    # is never implicitly deserialized from an arbitrary path.
    if isinstance(payload, tuple) and len(payload) == 5:
        return payload
    raise ValueError("checkpoint contains an unsupported NumPy RNG payload")


def _cpu_rng_tensor(value, label):
    """Validate and normalize a saved Torch RNG byte state to CPU."""
    if not torch.is_tensor(value) or value.ndim != 1 or value.dtype != torch.uint8:
        raise ValueError(f"{label} must be a one-dimensional torch.uint8 RNG tensor")
    return value.detach().to(device="cpu", dtype=torch.uint8).contiguous()


def capture_rng_state():
    """Capture every RNG that can alter a resumed training trajectory.

    ``_FREQ_RNG`` is particularly important: it chooses the stochastic
    frequency subset used by both train and validation, and its state is not
    represented by PyTorch's process-global RNG state.  The returned object is
    Torch-serializable and deliberately additive to old checkpoints.
    """
    state = {
        "version": 2,
        "python": random.getstate(),
        "numpy": _serialize_numpy_rng_state(),
        "torch_cpu": torch.get_rng_state(),
        "freq_cpu": _FREQ_RNG.get_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state, *, require_complete=False):
    """Restore :func:`capture_rng_state`; return ``True`` when exact restore succeeded.

    Historical checkpoints did not contain an RNG payload.  They remain
    loadable for legacy recipes, but an adaptive-v2 run refuses that weaker
    resume path because topology decisions would no longer be reproducible.
    """
    required = {"python", "numpy", "torch_cpu", "freq_cpu"}
    missing = required - set(state or {})
    if missing:
        if require_complete:
            raise ValueError(
                "adaptive-capacity-v2 resume requires a complete RNG payload; "
                f"checkpoint is missing {sorted(missing)}")
        print("NOTE: checkpoint has no complete RNG payload; legacy resume is state-compatible "
              "but not trajectory-identical.")
        return False
    try:
        python_state = state["python"]
        numpy_state = _decode_numpy_rng_state(state["numpy"])
        cpu_state = _cpu_rng_tensor(state["torch_cpu"], "torch_cpu")
        freq_state = _cpu_rng_tensor(state["freq_cpu"], "freq_cpu")
        saved_cuda = state.get("torch_cuda_all")
        if saved_cuda is not None:
            if not isinstance(saved_cuda, (list, tuple)):
                raise ValueError("torch_cuda_all must be a list of RNG byte tensors")
            saved_cuda = [_cpu_rng_tensor(value, f"torch_cuda_all[{index}]")
                          for index, value in enumerate(saved_cuda)]
            if not torch.cuda.is_available() or len(saved_cuda) != torch.cuda.device_count():
                raise ValueError(
                    "checkpoint CUDA RNG topology does not match this process "
                    f"(saved {len(saved_cuda)}, current "
                    f"{torch.cuda.device_count() if torch.cuda.is_available() else 0})")
        elif torch.cuda.is_available() and require_complete:
            raise ValueError("adaptive-capacity-v2 resume requires CUDA RNG state on a CUDA worker")
        # Validate every payload before mutating any generator, then restore.
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        _FREQ_RNG.set_state(freq_state)
        if saved_cuda is not None:
            torch.cuda.set_rng_state_all(saved_cuda)
    except (TypeError, ValueError, RuntimeError) as exc:
        if require_complete:
            raise ValueError(f"adaptive-capacity-v2 RNG restore failed: {exc}") from exc
        print(f"NOTE: checkpoint RNG payload could not be restored exactly: {exc}")
        return False
    return True


def adaptive_refinement_config(
    *, enabled, refine_every, probe_every, min_spatial_exposure,
    min_angular_exposure, spatial_fraction, angular_fraction,
    spatial_floor, angular_floor, cooldown_events, child_maturity_events,
    max_active, split_max_level, regularizer_normalization,
    regularizer_reference_active_count,
):
    """Canonical, checkpointed identity for an adaptive-capacity trajectory."""
    return {
        "version": "adaptive_capacity_v2.2" if enabled else None,
        "refine_every_epochs": int(refine_every),
        "probe_every_views": int(probe_every),
        "min_spatial_exposure": int(min_spatial_exposure),
        "min_angular_exposure": int(min_angular_exposure),
        "spatial_fraction": float(spatial_fraction),
        "angular_fraction": float(angular_fraction),
        "spatial_floor": float(spatial_floor),
        "angular_floor": float(angular_floor),
        "cooldown_events": int(cooldown_events),
        "child_maturity_events": int(child_maturity_events),
        "max_active": int(max_active),
        "split_max_level": int(split_max_level),
        "regularizer_normalization": str(regularizer_normalization),
        "regularizer_reference_active_count": int(regularizer_reference_active_count),
        # v2 enables a compute-only SH slice; fixed parameter/Adam allocation
        # remains deliberately unchanged for checkpoint compatibility.
        "compact_sh_eval": bool(enabled),
    }


def _dataset_role_identity(loader, role):
    """Return the exact view membership used by one adaptive training role.

    This is ordinary checkpoint metadata rather than a dataset hash/lock.  It
    prevents a recovery command from silently continuing a trajectory with a
    different train/validation split or source.  The sealed manifest path has
    one explicit relocation-aware comparison in
    :func:`_sealed_adaptive_training_resume_identity`; all ordinary recovery
    still compares this provenance literally.
    """
    dataset = loader.dataset
    identity = {
        "role": str(role),
        "dataset_type": f"{type(dataset).__module__}.{type(dataset).__qualname__}",
    }
    if hasattr(dataset, "indices"):
        indices = np.asarray(dataset.indices, dtype=np.int64)
        identity["view_indices"] = [int(index) for index in indices.tolist()]
        identity["source_path"] = os.path.abspath(str(getattr(dataset, "source_path", "")))
        return identity
    if hasattr(dataset, "file_paths"):
        identity["file_paths"] = [os.path.abspath(str(path)) for path in dataset.file_paths]
        return identity
    raise ValueError(
        "adaptive-capacity-v2 requires an identifiable train/validation dataset "
        f"for exact resume; {role} loader has {type(dataset).__qualname__} without indices or file_paths")


def _contract_value(value):
    """Canonical primitive/container form for checkpoint-comparable settings."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_contract_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _contract_value(value[key]) for key in sorted(value)}
    raise ValueError(f"adaptive-capacity-v2 checkpoint contract cannot encode {type(value).__name__}")


def _optimizer_scheduler_recipe(optimizer, scheduler):
    """Capture requested optimizer/scheduler knobs before a resume restores state.

    ``optimizer.load_state_dict`` overwrites these settings, which is correct
    for recovery but otherwise hides a changed command line.  Recording the
    fresh recipe first lets v2 reject that ambiguity rather than merely
    continuing under values different from those requested.
    """
    parameter_groups = []
    for group in optimizer.param_groups:
        parameter_groups.append({
            str(key): _contract_value(value)
            for key, value in group.items()
            if key != "params"
        })
    scheduler_settings = {}
    for name in ("T_0", "T_mult", "eta_min", "base_lrs"):
        if hasattr(scheduler, name):
            scheduler_settings[name] = _contract_value(getattr(scheduler, name))
    return {
        "optimizer_type": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
        "parameter_groups": parameter_groups,
        "scheduler_type": f"{type(scheduler).__module__}.{type(scheduler).__qualname__}",
        "scheduler_settings": scheduler_settings,
    }


def optimizer_scheduler_requested_recipe(
    optimizer, scheduler, *, requested_scene_lr=None, normalize_scene_scale=False,
):
    """Return the command-level optimizer identity used for strict recovery.

    ``normalize_scene_scale`` is an initialization-only gauge fix.  A fresh
    normalized run uses ``requested_scene_lr * scene_scale`` for its first
    (scene-coefficient) AdamW group; a recovery deliberately does *not* run
    normalization again, then restores that effective LR from the checkpoint's
    optimizer state.  Comparing the freshly constructed effective LR would
    therefore reject an otherwise identical recovery before state restoration.

    This contract records the user-requested pre-gauge scene LR instead, while
    retaining every other optimizer and scheduler setting verbatim.  The first
    parameter group is the scene group in every explicit-scene construction in
    :func:`main`; ``None`` keeps generic/non-scene callers byte-for-byte on the
    ordinary runtime recipe.  The saved optimizer/scheduler state remains the
    sole source of the *effective* LR after recovery.
    """
    recipe = _optimizer_scheduler_recipe(optimizer, scheduler)
    gauge = {
        "schema": "scene_lr_pre_gauge_v1",
        "enabled": bool(normalize_scene_scale),
        "requested_scene_lr": None,
    }
    if requested_scene_lr is not None:
        if not recipe["parameter_groups"]:
            raise ValueError("scene-LR resume contract needs a scene optimizer parameter group")
        requested_scene_lr = float(requested_scene_lr)
        first_group = recipe["parameter_groups"][0]
        # CosineAnnealingWarmRestarts records the initial group LR in both the
        # optimizer group and ``base_lrs``.  Canonicalize every copy that would
        # otherwise contain the fresh-run-only gauge multiplier.
        for key in ("lr", "initial_lr"):
            if key in first_group:
                first_group[key] = requested_scene_lr
        base_lrs = recipe["scheduler_settings"].get("base_lrs")
        if base_lrs is not None:
            if not isinstance(base_lrs, list) or not base_lrs:
                raise ValueError("scheduler base_lrs must describe the scene optimizer group")
            base_lrs[0] = requested_scene_lr
        gauge["requested_scene_lr"] = requested_scene_lr
    recipe["scene_scale_gauge"] = gauge
    return recipe


EXECUTION_CONTRACT_SCHEMA = "rift_checkpoint_execution_contract_v1"


def checkpoint_execution_contract(
    args,
    *,
    resolved_arr_dist,
    resolved_spacing,
    resolved_num_rx,
    resolved_num_tx,
    resolved_extent,
    resolved_granularity,
    resolved_sh_max_degree,
    init_scale,
    optimizer_requested_recipe,
):
    """Return an opt-in, derived checkpoint execution contract.

    Historical commands leave ``--execution-contract-label`` unset, so their
    checkpoint schema and resume behavior remain unchanged.  A new isolated
    lane can opt in to a compact record derived from the *actual parsed
    command*, rather than stamping a recipe beside an otherwise ambiguous
    generic checkpoint after training.  The sealed NPZ contract continues to
    own ordered roles/header access; this record covers the remaining
    trajectory-defining configuration.
    """

    label = args.execution_contract_label
    if label is None:
        return None
    if not isinstance(label, str) or not label.strip() or label != label.strip():
        raise ValueError("--execution-contract-label must be a nonempty trimmed string")
    physics = {
        "forward_operator": str(args.forward_operator),
        "range_model": str(args.range_model),
        "compute_dtype": str(args.compute_dtype),
        "phase_sign": float(args.phase_sign),
        "coordinate_source": (
            "npz_per_view_positions" if args.data_format == "npz" else "analytic_array_geometry"
        ),
        "num_rx": int(resolved_num_rx),
        "num_tx": int(resolved_num_tx),
        "point_chunk": int(args.point_chunk),
        "pair_chunk": int(args.pair_chunk),
    }
    if args.data_format != "npz":
        physics.update(
            {
                "array_distance_m": float(resolved_arr_dist),
                "element_spacing_m": float(resolved_spacing),
            }
        )
    return {
        "schema": EXECUTION_CONTRACT_SCHEMA,
        "label": label,
        "observation": {
            "data_format": str(args.data_format),
            "sealed_npz_protocol": bool(args.npz_sealed_protocol),
            "num_train": int(args.num_train),
            "num_validation": int(args.num_val),
            "num_reserved_test": int(args.num_test),
            "num_freq_wanted": int(args.num_freq_wanted),
            "validation_cap_axis": (
                None if args.val_cap_axis is None else [float(value) for value in args.val_cap_axis]
            ),
            "validation_from_tail": bool(args.val_from_tail),
        },
        "scene": {
            "representation": str(args.scene_repr),
            "extent_m": float(resolved_extent),
            "granularity": int(resolved_granularity),
            "initial_scale": float(init_scale),
            "backprojection_views": int(args.bp_init),
            "shell_init_radius_m": float(args.shell_init_radius),
            "normalize_scene_scale": bool(args.normalize_scene_scale),
            "sh_max_degree": int(resolved_sh_max_degree)
            if isinstance(resolved_sh_max_degree, (int, np.integer))
            else str(resolved_sh_max_degree),
            "sh_init_degree": int(args.sh_init_degree),
        },
        "physics": physics,
        "fit": {
            "epochs": int(args.epochs),
            "loss": str(args.loss),
            "step_every": int(args.step_every),
            "clip_grad_norm": float(args.clip_grad_norm),
            "learn_global_gain": not bool(args.no_learn_gain),
            "learning_rate": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "l1_weight": float(args.l1_weight),
            "sh_smooth_weight": float(args.sh_smooth_weight),
            "regularizer_normalization": str(args.regularizer_normalization),
            "adam_eps": float(args.adam_eps),
            "checkpoint_metric": str(args.checkpoint_metric),
            "seed": int(args.seed),
            "scheduler": {
                "t0": int(args.t0),
                "t_mult": int(args.t_mult),
                "eta_min": 1.0e-6,
            },
            "pruning": {
                "every": int(args.prune_every),
                "threshold": float(args.prune_threshold),
                "criterion": str(args.prune_criterion),
                "start_epoch": int(args.prune_start_epoch),
                "mode": str(args.prune_mode),
                "target_active": int(args.prune_target_active),
                "end_epoch": int(args.prune_end_epoch),
                "min_active": int(args.prune_min_active),
            },
            "view_weight_alpha": float(args.view_weight_alpha),
            "view_weight_max_ratio": float(args.view_weight_max_ratio),
            "magnitude_weight": float(args.mag_weight),
            "magnitude_warmup_epochs": int(args.mag_warmup_epochs),
            "resume_requires_full_state": bool(args.require_full_resume_state),
            "optimizer": {
                "name": "AdamW",
                "scene_learning_rate": float(args.lr),
                "gain_learning_rate": float(args.lr),
                "betas": [0.9, 0.999],
                "eps": float(args.adam_eps),
                "weight_decay": float(args.weight_decay),
            },
        },
    }


def validate_execution_contract(checkpoint, expected_execution_contract):
    """Reject opt-in continuation when its parsed configuration changed."""

    if expected_execution_contract is None:
        return
    if not isinstance(expected_execution_contract, Mapping):
        raise ValueError("expected execution contract must be a mapping")
    if checkpoint.get("execution_contract") != expected_execution_contract:
        raise ValueError(
            "checkpoint execution contract differs from the requested command; "
            "start a new output identity rather than changing the trajectory"
        )


def adaptive_training_contract(
    *, train_loader, validation_loader, num_freq_selected, loss_mode, w1, w2,
    l1_weight, sh_smooth_weight, regularizer_normalization,
    regularizer_reference_active_count, phase_sign, forward_operator_name,
    compute_dtype, data_format, op_kwargs, arr_dist, spacing, num_rx, num_tx,
    prune_every, prune_threshold, prune_criterion, prune_start_epoch,
    prune_mode, prune_target_active, prune_end_epoch, prune_min_active,
    grow_every, grow_threshold, grow_threshold_mode, grow_criterion,
    grow_tail_ratio, split_every, split_threshold, split_max_level,
    step_every, clip_grad_norm, checkpoint_metric, mag_weight,
    mag_warmup_epochs, view_weight_alpha, view_weight_max_ratio, scene_repr,
    gain, occlusion, val_cap_axis, num_epochs, optimizer, scheduler,
    optimizer_requested_recipe=None,
):
    """Canonical non-controller identity for an adaptive continuation.

    The adaptive controller configuration lives separately in
    :func:`adaptive_refinement_config`.  This contract records the remaining
    loss, physics, observation-role, and topology settings that would alter a
    resumed trajectory even if the controller itself were unchanged.
    """
    occlusion_contract = {
        "enabled": occlusion is not None,
    }
    if occlusion is not None:
        occlusion_contract.update({
            "key": str(occlusion["key"]),
            "n_steps": None if occlusion["n_steps"] is None else int(occlusion["n_steps"]),
            "step_frac": float(occlusion["step_frac"]),
            "point_chunk": int(occlusion["point_chunk"]),
            "learnable": any(parameter.requires_grad for parameter in occlusion["scale"].parameters()),
        })
    return {
        "version": "adaptive_training_contract_v2",
        "observations": {
            "data_format": str(data_format),
            "train": _dataset_role_identity(train_loader, "train"),
            "validation": _dataset_role_identity(validation_loader, "validation"),
        },
        "loss": {
            "loss_mode": str(loss_mode), "w1": float(w1), "w2": float(w2),
            "l1_weight": float(l1_weight), "sh_smooth_weight": float(sh_smooth_weight),
            "regularizer_normalization": str(regularizer_normalization),
            "regularizer_reference_active_count": int(regularizer_reference_active_count),
            "mag_weight": float(mag_weight), "mag_warmup_epochs": int(mag_warmup_epochs),
            "view_weight_alpha": float(view_weight_alpha),
            "view_weight_max_ratio": float(view_weight_max_ratio),
        },
        "physics": {
            "phase_sign": float(phase_sign),
            "forward_operator": str(forward_operator_name),
            "compute_dtype": str(compute_dtype),
            "operator_options": _contract_value(op_kwargs),
            "arr_dist": float(arr_dist), "spacing": float(spacing),
            "num_rx": int(num_rx), "num_tx": int(num_tx),
            "gain_enabled": gain is not None,
            "occlusion": occlusion_contract,
        },
        "topology": {
            "scene_repr": str(scene_repr),
            "prune_every": int(prune_every), "prune_threshold": float(prune_threshold),
            "prune_criterion": str(prune_criterion), "prune_start_epoch": int(prune_start_epoch),
            "prune_mode": str(prune_mode), "prune_target_active": int(prune_target_active),
            "prune_end_epoch": int(prune_end_epoch),
            # A zero CLI value means "end at --epochs", so compare the
            # resolved endpoint as well as the raw option.
            "prune_end_epoch_effective": int(prune_end_epoch or num_epochs),
            "prune_min_active": int(prune_min_active),
            "grow_every": int(grow_every), "grow_threshold": float(grow_threshold),
            "grow_threshold_mode": str(grow_threshold_mode), "grow_criterion": str(grow_criterion),
            "grow_tail_ratio": float(grow_tail_ratio),
            "split_every": int(split_every), "split_threshold": float(split_threshold),
            "split_max_level": int(split_max_level),
        },
        "optimizer_execution": {
            "step_every": int(step_every), "clip_grad_norm": float(clip_grad_norm),
            "checkpoint_metric": str(checkpoint_metric),
            "requested_recipe": (
                _contract_value(optimizer_requested_recipe)
                if optimizer_requested_recipe is not None
                else _optimizer_scheduler_recipe(optimizer, scheduler)
            ),
        },
        "validation": {
            "cap_axis": None if val_cap_axis is None else [float(axis) for axis in val_cap_axis],
        },
        "frequency_selection_count": int(num_freq_selected),
    }


def _sealed_adaptive_training_resume_identity(contract, sealed_npz_protocol_contract):
    """Remove only relocated-NPZ provenance from an adaptive contract.

    Adaptive-v2 historically treats the source path in each observation role
    as trajectory-defining.  Preserve that strict behavior for every legacy
    CSV/eager-NPZ continuation.  A manifest-bound sealed NPZ continuation is
    the narrow exception: its validated header, ordered role IDs, and
    response-access policy already describe the observations semantically,
    while the archive and manifest locations are intentionally provenance
    rather than resume pins.  Strip the two ``source_path`` values only for
    that already-selected sealed path; all role IDs, dataset types, physics,
    optimizer, and topology fields remain exact comparisons.

    This returns a fresh shallow hierarchy and never mutates checkpoint
    metadata, so the original paths remain visible in saved provenance.
    """
    if sealed_npz_protocol_contract is None:
        return contract
    if not isinstance(contract, Mapping):
        return contract
    observations = contract.get("observations")
    if not isinstance(observations, Mapping) or observations.get("data_format") != "npz":
        return contract

    normalized = dict(contract)
    normalized_observations = dict(observations)
    normalized["observations"] = normalized_observations
    for role in ("train", "validation"):
        role_identity = normalized_observations.get(role)
        if isinstance(role_identity, Mapping):
            role_identity = dict(role_identity)
            role_identity.pop("source_path", None)
            normalized_observations[role] = role_identity
    return normalized


def validate_adaptive_resume_config(checkpoint, expected, expected_training_contract=None,
                                    sealed_npz_protocol_contract=None):
    """Reject a resume that would change the adaptive controller's meaning.

    ``sealed_npz_protocol_contract`` is supplied only by the already-validated
    manifest-bound NPZ continuation path.  It allows an archive relocation to
    remain provenance-only there; nonsealed recovery keeps its historical
    whole-contract (including source-path) comparison.
    """
    saved_enabled = bool(checkpoint.get("adaptive_capacity_v2", False))
    expected_enabled = expected.get("version") is not None
    if saved_enabled != expected_enabled:
        raise ValueError(
            "checkpoint and requested recipe disagree about --adaptive-capacity-v2; "
            "start a new checkpoint directory rather than changing topology policy in place")
    if not expected_enabled:
        return
    saved = checkpoint.get("adaptive_refinement")
    if not isinstance(saved, dict):
        raise ValueError("adaptive-capacity-v2 checkpoint lacks its adaptive_refinement identity")
    mismatches = {
        key: (saved.get(key), expected[key])
        for key in expected
        if saved.get(key) != expected[key]
    }
    if mismatches:
        detail = "; ".join(f"{key}: saved={old!r}, requested={new!r}"
                           for key, (old, new) in mismatches.items())
        raise ValueError(
            "adaptive-capacity-v2 resume would change a trajectory-defining setting: " + detail)
    if expected_training_contract is None:
        raise ValueError("adaptive-capacity-v2 resume requires an expected training contract")
    saved_training_contract = checkpoint.get("adaptive_training_contract")
    if not isinstance(saved_training_contract, dict):
        raise ValueError(
            "adaptive-capacity-v2.2 resume requires its complete training contract; "
            "start a new checkpoint directory rather than changing the scientific recipe in place")
    saved_training_identity = _sealed_adaptive_training_resume_identity(
        saved_training_contract, sealed_npz_protocol_contract)
    expected_training_identity = _sealed_adaptive_training_resume_identity(
        expected_training_contract, sealed_npz_protocol_contract)
    if saved_training_identity != expected_training_identity:
        raise ValueError(
            "adaptive-capacity-v2 resume would change its loss, physics, observation split, "
            "or optimizer/topology execution contract")
    scene_state = checkpoint.get("model_state_dict")
    if not isinstance(scene_state, dict):
        raise ValueError("adaptive-capacity-v2 checkpoint lacks a model_state_dict")
    required_scene_buffers = {
        "support_min", "support_max", "support_bounds_enabled",
        "compact_sh_eval_enabled", "refine_birth_event",
    }
    missing_buffers = required_scene_buffers - set(scene_state)
    if missing_buffers:
        raise ValueError(
            "adaptive-capacity-v2.2 resume requires its support/compact/maturity scene state; "
            f"checkpoint is missing {sorted(missing_buffers)}")
    if not (bool(scene_state["support_bounds_enabled"].item())
            and bool(scene_state["compact_sh_eval_enabled"].item())):
        raise ValueError(
            "adaptive-capacity-v2.2 checkpoint disables immutable support bounds or "
            "compact active-SH evaluation")
    rng_state = checkpoint.get("rng_state")
    rng_version = rng_state.get("version") if isinstance(rng_state, dict) else None
    if not isinstance(rng_version, int) or rng_version != 2:
        raise ValueError(
            "adaptive-capacity-v2.2 resume requires safe RNG payload version 2; "
            "start a new checkpoint directory rather than resuming a pre-v2.2 trajectory")


def build_dataloaders(data_dir, num_train, num_val, num_test, device):
    selected_files_training = list_and_select_files(data_dir, num_files=num_train)
    selected_files_validation = list_and_select_files(
        data_dir, num_files=num_val, exclude_files=selected_files_training
    )
    used_data = selected_files_training + selected_files_validation
    selected_files_test = list_and_select_files(data_dir, num_files=num_test, exclude_files=used_data)

    training_dataset = CSVSimulationDataset(selected_files_training, device=device)
    validation_dataset = CSVSimulationDataset(selected_files_validation, device=device)
    test_dataset = CSVSimulationDataset(selected_files_test, device=device)

    training_data_loader = DataLoader(training_dataset, batch_size=1)
    validation_data_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return training_data_loader, validation_data_loader, test_data_loader


_SEALED_NPZ_PROTOCOL_SCHEMA = "rift_npz_sealed_protocol_v1"
_SEALED_NPZ_RESUME_IDENTITY_FIELDS = (
    "schema",
    "version",
    "data_format",
    "response_shape",
    "response_dtype",
    "split_strategy",
    "role_ids",
    "response_access",
)


def _manifest_nonnegative_int(value, label):
    """Return one JSON integer without accepting bools or silent coercions."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} must be an integer, got {value!r}")
    value = int(value)
    if value < 0:
        raise ValueError(f"{label} must be non-negative, got {value}")
    return value


def _sealed_manifest_indices(split, key, expected_count, num_views):
    """Validate one ordered role list before any NPZ response is opened."""
    values = split.get(key)
    if not isinstance(values, list):
        raise ValueError(f"sealed NPZ manifest split.{key} must be a JSON list")
    if len(values) != expected_count:
        raise ValueError(
            f"sealed NPZ manifest split.{key} has {len(values)} IDs, expected {expected_count}")
    indices = [_manifest_nonnegative_int(value, f"sealed NPZ manifest split.{key}")
               for value in values]
    if any(value >= num_views for value in indices):
        raise ValueError(
            f"sealed NPZ manifest split.{key} contains an ID outside [0, {num_views})")
    if len(set(indices)) != len(indices):
        raise ValueError(f"sealed NPZ manifest split.{key} contains duplicate IDs")
    return indices


def _sealed_manifest_count(split, key):
    if key not in split:
        raise ValueError(f"sealed NPZ manifest is missing split.{key}")
    return _manifest_nonnegative_int(split[key], f"sealed NPZ manifest split.{key}")


def _load_sealed_npz_protocol_contract(npz_path, role_manifest_path, *,
                                       num_train, num_val, num_test,
                                       num_tx=None, num_rx=None, tx_indices=None, rx_indices=None):
    """Bind manifest roles from metadata/header data before response access.

    This is deliberately a small, opt-in protocol rather than a replacement for
    the historical seed/count loader.  It records ordered roles directly in a
    checkpoint-comparable contract; it does not add a source hash, a lock, or
    any other second coordination mechanism.  The existing lazy reader
    authorizes only train/validation IDs for materialization.  It can consume
    compressed archive bytes while scanning toward an authorized view, but it
    never creates a reserved-test response array, tensor, or DataLoader.
    """
    manifest_path = os.path.abspath(os.fspath(role_manifest_path))
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, Mapping):
        raise ValueError("sealed NPZ role manifest must be a JSON object")
    if manifest.get("schema_version") != 1:
        raise ValueError("sealed NPZ role manifest requires schema_version 1")
    name = manifest.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("sealed NPZ role manifest requires a non-empty name")
    dataset = manifest.get("dataset")
    split = manifest.get("split")
    if not isinstance(dataset, Mapping) or not isinstance(split, Mapping):
        raise ValueError("sealed NPZ role manifest requires object-valued dataset and split")

    from rift.rift_dataset import DATASET_ID, load_object_contract
    if (dataset.get("dataset_id") == DATASET_ID and "engineering_subset" not in manifest
            and (manifest.get("antenna_selection") is not None or num_train != split.get("num_train")
                 or any(v is not None for v in (num_tx, num_rx, tx_indices, rx_indices)))):
        arrays, contract = load_object_contract(npz_path, manifest_path, num_train=num_train,
            num_tx=num_tx, num_rx=num_rx, tx_indices=tx_indices, rx_indices=rx_indices)
        if (len(contract["role_ids"]["validation"]), len(contract["role_ids"]["reserved_test"])) != (num_val, num_test):
            raise ValueError("sealed NPZ manifest role counts disagree with the requested command")
        return arrays, contract

    engineering = manifest.get("engineering_subset")
    if isinstance(engineering, Mapping) and engineering.get("schema") == "rift_dataset_sugavanam_ertin_smoke_v1":
        if (num_train, num_val, num_test) != (16, 16, 1000):
            raise ValueError("RIFT dataset SE smoke requires exactly 16/16/1000 requested roles")
        from rift.sugavanam_ertin_b7873200_real_smoke import load_collection_smoke_inputs
        return load_collection_smoke_inputs(npz_path, manifest_path)

    # This lazy header/metadata load precedes all split parsing that could
    # authorize response rows.  ``response`` remains None until the restricted
    # datasets below ask for their own train/validation source IDs.
    arrays = load_npz_arrays(npz_path, load_response=False)
    from rift.rift_dataset import validate_manifest_object
    dataset_identity = validate_manifest_object(manifest, arrays["meta"])
    response_shape = tuple(int(value) for value in arrays["response_shape"])
    response_dtype = str(arrays["response_dtype"])
    num_views = npz_response_num_views(arrays)

    manifest_num_views = _manifest_nonnegative_int(
        dataset.get("num_views"), "sealed NPZ manifest dataset.num_views")
    if manifest_num_views != num_views:
        raise ValueError(
            "sealed NPZ manifest dataset.num_views disagrees with the NPZ response header: "
            f"manifest={manifest_num_views}, header={num_views}")
    manifest_shape = dataset.get("response_shape")
    if not isinstance(manifest_shape, list):
        raise ValueError(
            "sealed NPZ manifest dataset.response_shape disagrees with the NPZ response header")
    manifest_shape = tuple(
        _manifest_nonnegative_int(value, "sealed NPZ manifest dataset.response_shape")
        for value in manifest_shape
    )
    if manifest_shape != response_shape:
        raise ValueError(
            "sealed NPZ manifest dataset.response_shape disagrees with the NPZ response header")
    if dataset.get("response_dtype") != response_dtype:
        raise ValueError(
            "sealed NPZ manifest dataset.response_dtype disagrees with the NPZ response header")

    if split.get("complete_partition") is not True:
        raise ValueError("sealed NPZ manifest must declare split.complete_partition=true")
    if split.get("test_sealed") is not True:
        raise ValueError("sealed NPZ manifest must declare split.test_sealed=true")
    strategy = split.get("strategy")
    if not isinstance(strategy, str) or not strategy.strip():
        raise ValueError("sealed NPZ manifest requires a non-empty split.strategy")
    train_count = _sealed_manifest_count(split, "num_train")
    validation_count = _sealed_manifest_count(split, "num_validation")
    test_count = _sealed_manifest_count(split, "num_test")
    if (train_count, validation_count, test_count) != (int(num_train), int(num_val), int(num_test)):
        raise ValueError(
            "sealed NPZ manifest role counts disagree with the requested command: "
            f"manifest={(train_count, validation_count, test_count)}, "
            f"requested={(int(num_train), int(num_val), int(num_test))}")
    if not (train_count and validation_count and test_count):
        raise ValueError("sealed NPZ protocol requires non-empty train, validation, and reserved-test roles")

    train_ids = _sealed_manifest_indices(split, "train_indices", train_count, num_views)
    validation_ids = _sealed_manifest_indices(
        split, "validation_indices", validation_count, num_views)
    reserved_test_ids = _sealed_manifest_indices(split, "test_indices", test_count, num_views)
    if "unused_indices" in split or "num_unused" in split:
        unused_count = _sealed_manifest_count(split, "num_unused")
        unused_ids = _sealed_manifest_indices(split, "unused_indices", unused_count, num_views)
        if unused_ids and split.get("unused_sealed") is not True:
            raise ValueError("sealed NPZ manifest must declare split.unused_sealed=true for unused IDs")
    else:
        unused_ids = []

    role_ids = {
        "train": train_ids,
        "validation": validation_ids,
        "reserved_test": reserved_test_ids,
        "unused": unused_ids,
    }
    all_ids = [view_id for role in role_ids.values() for view_id in role]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("sealed NPZ manifest roles overlap")
    if set(all_ids) != set(range(num_views)):
        raise ValueError(
            "sealed NPZ manifest claims a complete partition but its roles do not cover every NPZ view ID")

    contract = {
        "schema": _SEALED_NPZ_PROTOCOL_SCHEMA,
        "version": 1,
        "data_format": "npz",
        "source_path": os.path.abspath(os.fspath(npz_path)),
        "response_shape": list(response_shape),
        "response_dtype": response_dtype,
        "role_manifest_path": manifest_path,
        "role_manifest_name": name.strip(),
        "split_strategy": strategy,
        "role_ids": role_ids,
        "response_access": {
            "train_materialized": True,
            "validation_materialized": True,
            "reserved_test_materialized": False,
            "unused_materialized": False,
        },
    }
    if dataset_identity is not None:
        contract["dataset_identity"] = dataset_identity
        if "training_selection" in manifest:
            contract["training_selection"] = manifest["training_selection"]
    return arrays, contract


def _sealed_npz_resume_identity(contract, label):
    """Return the semantic part of a sealed contract, excluding provenance."""
    if not isinstance(contract, Mapping):
        raise ValueError(f"{label} sealed_npz_protocol_contract must be an object")
    missing = [field for field in _SEALED_NPZ_RESUME_IDENTITY_FIELDS if field not in contract]
    if missing:
        raise ValueError(f"{label} sealed_npz_protocol_contract is missing {missing}")
    # ``source_path`` and the manifest's path/name describe where the command
    # happened to find its data.  They are useful checkpoint provenance, but
    # deliberately not a path/hash pin: a corrected or relocated archive with
    # the same validated header and frozen roles is the same continuation.
    return {
        field: contract[field]
        for field in _SEALED_NPZ_RESUME_IDENTITY_FIELDS
    } | ({"dataset_identity": contract["dataset_identity"]}
         if "dataset_identity" in contract else {}) | {
            key: contract[key] for key in ("antenna_selection", "source_geometry_sha256", "source_response_shape")
            if key in contract}


def _validate_saved_sealed_npz_protocol_contract(saved_contract, expected_contract):
    """Check both directions of a sealed/legacy checkpoint transition."""
    if expected_contract is None:
        if saved_contract is not None:
            raise ValueError(
                "a checkpoint with sealed_npz_protocol_contract cannot resume through a legacy "
                "or eager data path; pass --npz-sealed-protocol and its original role manifest")
        return
    if not isinstance(saved_contract, Mapping):
        raise ValueError(
            "sealed NPZ resume requires a checkpoint with its complete sealed_npz_protocol_contract; "
            "start a new checkpoint directory rather than mixing legacy and sealed trajectories")
    saved_identity = _sealed_npz_resume_identity(saved_contract, "checkpoint")
    expected_identity = _sealed_npz_resume_identity(expected_contract, "requested")
    if saved_identity != expected_identity:
        raise ValueError(
            "sealed NPZ resume would change its schema/header, ordered roles, or response-access policy")


def build_sealed_npz_dataloaders(npz_path, role_manifest_path, *,
                                  num_train, num_val, num_test,
                                  resume_sealed_npz_protocol_contract=None,
                                  num_tx=None, num_rx=None, tx_indices=None, rx_indices=None):
    """Build manifest-bound train/validation loaders without a test loader.

    The role manifest is validated against metadata and the response header
    before the lazy reader is narrowed to train plus validation IDs.  The
    returned ``None`` is intentional: the training entrypoint has no capability
    to materialize or iterate the reserved-test role.
    """
    arrays, contract = _load_sealed_npz_protocol_contract(
        npz_path,
        role_manifest_path,
        num_train=num_train,
        num_val=num_val,
        num_test=num_test, num_tx=num_tx, num_rx=num_rx, tx_indices=tx_indices, rx_indices=rx_indices,
    )
    # Validate a continuation's exact role/header contract while the NPZ is
    # still metadata-only.  Constructing PecSphereNPZDataset below turns the
    # authorized train/validation rows into tensors, so a mismatch must stop
    # here rather than after an avoidable response read.
    if resume_sealed_npz_protocol_contract is not None:
        _validate_saved_sealed_npz_protocol_contract(
            resume_sealed_npz_protocol_contract, contract)
    roles = contract["role_ids"]
    authorized_ids = np.asarray(roles["train"] + roles["validation"], dtype=np.int64)
    restricted_arrays = restrict_npz_response_views(arrays, authorized_ids)
    training_dataset = PecSphereNPZDataset(
        restricted_arrays, roles["train"], source_path=npz_path)
    validation_dataset = PecSphereNPZDataset(
        restricted_arrays, roles["validation"], source_path=npz_path)
    training_data_loader = DataLoader(training_dataset, batch_size=1)
    validation_data_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
    return training_data_loader, validation_data_loader, None, contract


def validate_sealed_npz_resume_contract(checkpoint, expected_contract):
    """Reject a recovery command whose sealed roles/protocol changed."""
    _validate_saved_sealed_npz_protocol_contract(
        checkpoint.get("sealed_npz_protocol_contract"), expected_contract)


def reshape_measured_cubes(magnitude_tensor, phase_tensor, device, num_tx, num_rx):
    """Flat CSV channel columns are ordered Tx-outer/Rx-inner (verified against
    the real .frtm-derived channel names: all Rx values are listed before
    Tx advances). view(-1, num_tx, num_rx) matches that true structure;
    permute(2,1,0) then reorders to [Rx, Tx, freq], matching
    forward_operator_lessparallel's own [Rx, Tx] output convention. The
    old view(-1, num_rx, num_tx).permute(1,2,0) silently mismatched Rx/Tx
    pairs -- harmless-looking (no shape error) for a square 16x16 array,
    but produces scrambled (not just transposed) results for a non-square
    array like B787's 16 Tx x 15 Rx, which is how this was caught.
    """
    magnitude_cube = magnitude_tensor.squeeze(0).to(device).view(-1, num_tx, num_rx).permute(2, 1, 0)
    phase_cube = phase_tensor.squeeze(0).to(device).view(-1, num_tx, num_rx).permute(2, 1, 0)
    return magnitude_cube, phase_cube


def backprojection_init(model, train_loader, device, num_freq_selected,
                        arr_dist, spacing, num_rx, num_tx, max_viewpoints=16,
                        phase_sign=1.0, forward_operator_name="brute", compute_dtype=torch.float64,
                        data_format="csv", op_kwargs=None):
    """Initialize an explicit grid scene at the scaled matched-filter /
    backprojection image A^H s -- the classical coherent SAR image on the
    voxel grid, which is also the gradient direction of the complex-MSE
    objective at an empty scene. The global scale is then set in closed form
    (alpha = <A b, s> / ||A b||^2), so training starts at the best scalar
    multiple of the backprojection image (measured ~20-30% relative MSE on
    the AEDT sphere data) instead of at the empty scene (100%). grid_sh
    scenes receive the image in their isotropic (l=0) SH coefficient.

    data_format="npz": each batch carries exact rx_pos/tx_pos tensors (see
    rift/npz_dataset.py) instead of a (dtheta, dphi) pair to feed
    get_array_pos -- ground-truth geometry, no analytic-convention
    assumption. arr_dist/spacing are unused in that case.

    Multi-GPU (scene-sharded): every rank backprojects onto ITS OWN voxels,
    which needs no communication -- A^H s is elementwise in the scatterer
    index. Only the closed-form global scale alpha = <A b, s> / ||A b||^2
    does: A b is a sum over all ranks' points, so the rendered P and the
    two accumulators are all-reduced below. The resulting init is bitwise
    the same scene as the single-GPU path up to summation order.
    """
    op_kwargs = op_kwargs or {}
    is_sh_grid_scene = isinstance(model, SHVoxelGridScene)
    is_sharded_scene = isinstance(model, ShardedSHVoxelGridScene)
    is_point_scene = isinstance(model, AdaptivePointSHScene)
    if is_point_scene:
        # capacity-padded point scenes carry inactive spare slots (see
        # sparse_scene.py split()); backproject onto active points only so
        # inactive slots keep their all-zero invariant
        bp_mask = model.active_mask
        pos = model.grid_positions.reshape(-1, 3)[bp_mask]
    else:
        pos = model.grid_positions.reshape(-1, 3)
    b = torch.zeros(pos.shape[0], dtype=torch.cfloat, device=device)
    views = []
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i >= max_viewpoints:
                break
            if data_format == "npz":
                freqs_tensor, dphi_tensor, dtheta_tensor, magnitude_tensor, phase_tensor, rx_pos_batch, tx_pos_batch = batch
            else:
                freqs_tensor, dphi_tensor, dtheta_tensor, magnitude_tensor, phase_tensor = batch
            magnitude_cube, phase_cube = reshape_measured_cubes(
                magnitude_tensor, phase_tensor, device, num_tx, num_rx
            )
            freqs_tensor = freqs_tensor.squeeze(0).to(device)
            freq_indices = select_freq_indices(freqs_tensor.shape[0], num_freq_selected, device)
            selected_freqs = freqs_tensor[freq_indices]
            S_meas = torch.polar(
                magnitude_cube[:, :, freq_indices], phase_cube[:, :, freq_indices]
            ).permute(2, 0, 1)  # [nf, Rx, Tx], matching the forward operator's output layout
            k_vector_full = get_kvector(freqs_tensor, cc)
            k_vector = get_kvector(selected_freqs, cc)
            if data_format == "npz":
                rx_pos = rx_pos_batch.squeeze(0).to(device)
                tx_pos = tx_pos_batch.squeeze(0).to(device)
            else:
                rx_pos, tx_pos = get_array_pos(
                    dtheta_tensor.to(device), dphi_tensor.to(device), arr_dist, spacing, num_rx, num_tx, device
                )
            if forward_operator_name == "range":
                b += range_adjoint_operator(
                    freqs_tensor, k_vector_full, rx_pos, tx_pos, pos, S_meas,
                    phase_sign=phase_sign, freq_indices=freq_indices, compute_dtype=compute_dtype,
                    **op_kwargs,
                )
                views.append((freqs_tensor, k_vector_full, freq_indices, rx_pos, tx_pos, S_meas))
            else:
                b += adjoint_operator_lessparallel(
                    selected_freqs, k_vector, rx_pos, tx_pos, pos, S_meas, phase_sign=phase_sign,
                    **op_kwargs,
                )
                views.append((selected_freqs, k_vector, None, rx_pos, tx_pos, S_meas))

        num = torch.zeros((), dtype=torch.cfloat, device=device)
        den = torch.zeros((), device=device)
        for freqs_for_view, k_vector, freq_indices, rx_pos, tx_pos, S_meas in views:
            if forward_operator_name == "range":
                P = range_forward_operator(
                    freqs_for_view, k_vector, rx_pos, tx_pos, pos, b,
                    phase_sign=phase_sign, freq_indices=freq_indices, compute_dtype=compute_dtype,
                    **op_kwargs,
                )
            else:
                P = forward_operator_lessparallel(
                    freqs_for_view, k_vector, rx_pos, tx_pos, pos, b,
                    artificial_gain=1.0, p_spectrum=None,
                    omega_scaling="unity", center_freq_hz=None,
                    phase_sign=phase_sign, **op_kwargs,
                )
            P = all_reduce_sum(P)  # partial renders -> the full-scene A b
            num += (P.conj() * S_meas).sum()
            den += (P.real ** 2 + P.imag ** 2).sum()
        w = (num / den.clamp_min(1e-30)) * b

        if is_sh_grid_scene:
            g = model.granularity
            y00 = 0.5 / np.sqrt(np.pi)  # real SH basis value of the l=0 term
            model.w_re[..., 0].copy_((w.real / y00).view(g, g, g))
            model.w_im[..., 0].copy_((w.imag / y00).view(g, g, g))
        elif is_sharded_scene:
            # flat [n_local, n_basis]: this rank's own voxels, same DC term
            y00 = 0.5 / np.sqrt(np.pi)
            model.w_re[:, 0].copy_(w.real / y00)
            model.w_im[:, 0].copy_(w.imag / y00)
        elif is_point_scene:
            y00 = 0.5 / np.sqrt(np.pi)
            model.w_re[bp_mask, 0] = w.real / y00
            model.w_im[bp_mask, 0] = w.imag / y00
        else:
            g = model.granularity
            model.w_re.copy_(w.real.view(g, g, g))
            model.w_im.copy_(w.imag.view(g, g, g))
    w_max = w.abs().max() if w.numel() else torch.zeros((), device=device)
    print(f"Backprojection init: {len(views)} viewpoints, |w| max = {all_reduce_max(w_max).item():.3e}, "
          f"scale alpha = {complex(num / den.clamp_min(1e-30)):.3e}")


def regularization_loss(
    model,
    l1_weight,
    sh_smooth_weight,
    gain=None,
    return_terms=False,
    normalization="active_mean",
    reference_active_count=None,
):
    """Prior terms added to every optimizer step's objective (explicit scene
    representations only; returns None for the MLP or when both weights are 0).

    l1_weight * |g| * mean_over_active(|w|): group sparsity -- one smoothed
    L2-norm per voxel/point across its SH coefficients, L1 across entries.
    Pushes the optimizer toward FEW nonzero scatterers instead of the diffuse
    backprojection-like solution, attacking both the speckle overfit behind
    the >100% val rel-MSE (a fragmented fast-phase field only sums coherently
    at train geometries) and the ~2x shell-width excess. The +eps inside the
    sqrt keeps the gradient exactly 0 (not NaN) at w=0, so pruned/empty
    entries are safe.

    sh_smooth_weight * |g|^2 * mean_over_active(sum_lm l(l+1)|c_lm|^2):
    degree-weighted angular energy, equivalently the spherical
    Laplace-Beltrami seminorm. Degree 0 is free and progressively higher SH
    bands cost more, so narrow specular lobes survive only where the data
    genuinely demands them. This is exposed as ``--sh-degree-weight``;
    ``--sh-smooth-weight`` remains an exact backward-compatible CLI alias.
    Locked (never-unlocked) coefficients are exactly zero and contribute
    nothing.

    The |g| / |g|^2 factors (the differentiable magnitude of the learnable
    GlobalComplexGain) make both terms scale-INVARIANT: without them,
    training could shrink every weight by c and inflate the gain by c --
    identical predictions, arbitrarily small penalty -- so the prior would
    be minimized by rescaling instead of by sparsifying. With them the
    penalty depends only on the product g*w (the effective reflectivity in
    measurement units), which rescaling cannot change.

    ``normalization='active_mean'`` is the historical default: means are over
    ACTIVE entries, so a given weight transfers between granularities (g24 vs
    g48).  ``'fixed_initial'`` instead divides by an immutable initial active
    count.  The latter is mandatory for an adaptive split recipe with a
    nonzero prior: adding seven zero-weight siblings must not dilute the
    unchanged heir's regularization by 8x.  It is a new, explicitly recorded
    recipe choice; historical runs keep their exact active-mean behavior.

    Multi-GPU: the mean must be over the GLOBAL active set, not each rank's
    shard -- a per-shard mean would make the penalty depend on how the
    voxels happen to be distributed. Both terms are therefore computed as
    local SUMS, all-reduced (differentiably: d(total)/d(local term) = 1),
    and divided by the global active count. Every rank ends up with the
    same scalar and with gradients only into its own weights. A rank whose
    shard is entirely pruned still contributes its (zero) sum, so all ranks
    reach the collective -- hence no early return once the group is up.
    """
    empty_result = (None, {}) if return_terms else None
    if l1_weight <= 0 and sh_smooth_weight <= 0:
        return empty_result
    if normalization not in {"active_mean", "fixed_initial"}:
        raise ValueError("regularizer normalization must be 'active_mean' or 'fixed_initial'")
    is_sh = isinstance(model, (SHVoxelGridScene, ShardedSHVoxelGridScene, AdaptivePointSHScene))
    if not is_sh and not isinstance(model, VoxelGridScene):
        return empty_result
    mask = model.active_mask.reshape(-1)
    if not is_dist() and not mask.any():
        return empty_result
    n_active = all_reduce_int(int(mask.sum().item()), device=model.w_re.device)
    if n_active == 0:
        return empty_result
    if normalization == "fixed_initial":
        if reference_active_count is None or int(reference_active_count) < 1:
            raise ValueError("fixed_initial regularization needs reference_active_count >= 1")
        normalizer = int(reference_active_count)
    else:
        normalizer = n_active
    gmag = torch.exp(gain.log_mag) if gain is not None else None
    w_re = model.w_re.reshape(-1, model.w_re.shape[-1])[mask] if is_sh else model.w_re.reshape(-1)[mask]
    w_im = model.w_im.reshape(-1, model.w_im.shape[-1])[mask] if is_sh else model.w_im.reshape(-1)[mask]
    sq = w_re ** 2 + w_im ** 2
    reg = None
    terms = {}
    if l1_weight > 0:
        group_sq = sq.sum(dim=-1) if is_sh else sq
        if normalization == "fixed_initial":
            # Keep a zero group's zero derivative without giving every newly
            # created zero sibling a tiny additive objective.  The historical
            # active-mean branch below remains bit-for-bit unchanged.
            group_norm = torch.sqrt(group_sq + 1e-24) - 1e-12
        else:
            group_norm = torch.sqrt(group_sq + 1e-24)
        l1 = l1_weight * all_reduce_sum_grad(group_norm.sum()) / normalizer
        terms["l1"] = l1 * gmag if gmag is not None else l1
        reg = terms["l1"]
    if sh_smooth_weight > 0 and is_sh:
        lap = (model.basis_degree * (model.basis_degree + 1)).to(sq.dtype)
        smooth = sh_smooth_weight * all_reduce_sum_grad((sq * lap).sum(dim=-1).sum()) / normalizer
        if gmag is not None:
            smooth = smooth * gmag ** 2
        terms["sh_degree"] = smooth
        reg = smooth if reg is None else reg + smooth
    return (reg, terms) if return_terms else reg


def shell_init(model, radius, extent, granularity):
    """Initialize the scene as an ideal thin isotropic shell: |w| = 1, zero
    phase, on entries whose anchor lies within half a cell of the given
    radius; everything else zero. Alternative to backprojection_init for
    testing whether the +8-10cm BP-inherited shell bias is an INIT artifact
    (see EXPERIMENT_MANAGER_HANDOFF.md 2026-07-13, shell-misfit experiment):
    the ideal shell is unbiased by construction, so if training holds it
    there, BP's outward halo -- not the objective -- owns the bias. The
    learnable GlobalComplexGain warm-starts the absolute scale/phase from the
    first viewpoint, so unit weights are sufficient here.
    """
    is_sh_grid_scene = isinstance(model, SHVoxelGridScene)
    is_sharded_scene = isinstance(model, ShardedSHVoxelGridScene)
    is_point_scene = isinstance(model, AdaptivePointSHScene)
    with torch.no_grad():
        if is_point_scene:
            pos = model.anchors
            tol = model.cell_half[:, 0]
            sel = model.active_mask & ((pos.norm(dim=-1) - radius).abs() <= tol)
        else:
            pos = model.grid_positions.reshape(-1, 3)
            tol = extent / granularity
            sel = (pos.norm(dim=-1) - radius).abs() <= tol
        y00 = 0.5 / np.sqrt(np.pi)  # real SH basis value of the l=0 term
        model.w_re.zero_()
        model.w_im.zero_()
        if is_point_scene or is_sharded_scene:
            model.w_re[sel, 0] = 1.0 / y00
        elif is_sh_grid_scene:
            g = model.granularity
            model.w_re[..., 0][sel.view(g, g, g)] = 1.0 / y00
        else:
            g = model.granularity
            model.w_re[sel.view(g, g, g)] = 1.0
    n_sel = all_reduce_int(int(sel.sum().item()), device=model.w_re.device)
    print(f"Shell init: {n_sel} entries within half a cell of r={radius:.3f}m set to |w|=1 "
          f"(gain warm-start will set the absolute scale)")
    if n_sel == 0:
        raise ValueError(f"shell init selected 0 entries -- radius {radius} outside the grid, "
                         f"or extent/granularity mismatch")


def apply_occlusion(model, scatterer_weights, rx_pos, tx_pos, occlusion):
    """Multiply the rendered weights by this viewpoint's two-way transmittance.

    Sits between the scene and the operator, so BOTH operators (`brute` and
    `range`) are untouched: the factor is frequency-independent and -- by the
    small-aperture argument in rift/occlusion.py -- pair-independent, so it
    folds into the complex weight and the range factorization survives.

    Returns the weights unchanged when occlusion is off. zeta -> 0 recovers
    that case exactly, which is why an occlusion arm can only match or beat
    its own baseline on the training objective.
    """
    if occlusion is None:
        return scatterer_weights
    t2 = view_transmittance(
        model, occlusion["scale"], array_phase_centre(rx_pos, tx_pos),
        key=occlusion["key"], n_steps=occlusion["n_steps"],
        step_frac=occlusion["step_frac"], point_chunk=occlusion["point_chunk"],
    )
    return scatterer_weights * t2.to(scatterer_weights.dtype)


def compute_view_weights(train_loader, alpha, device, max_weight_ratio=0.0):
    """Per-viewpoint weights ``w_v ~ p_v^-alpha`` for the training objective.

    ``p_v = sum_{rx,tx,f} |S_meas|^2`` is the measured power of viewpoint v over
    ALL frequencies and Tx/Rx pairs -- a fixed property of the view, so the
    weight does not move with the per-epoch random frequency subset.

    WHY (measured 2026-08-09 on the B787 held-out set): the data-fit objective
    is an unweighted sum of per-view squared residuals, so a view contributes in
    proportion to its power. The B787 view sphere is dominated by specular
    flashes -- the brightest 20% of views carry 79.3% of the power -- which puts
    the effective sample size at

        N_eff = (sum_v p_v)^2 / sum_v p_v^2 = 286 of 1800 training views,

    BELOW the dataset's own 558-view angular-Nyquist estimate. Yet those bright
    views hold only 51.1% of the global error mass: the dim 80% carry 20.7% of
    the power but 48.9% of the error, and predicting them perfectly would take
    the reported global rel-MSE from 25.05% to 12.80%. So this is not a change
    of metric -- roughly half the headroom in the canonical power-weighted
    global rel-MSE sits in views the objective currently under-weights.

    alpha interpolates between the two extremes and MUST be swept, not assumed:
    alpha = 0 reproduces the historical objective exactly, alpha = 1 fully
    normalizes each view. On the model-free spherical-harmonic interpolator of
    the same data (``scripts/eval_bandlimit_oracle.py``) the GLOBAL held-out
    error is not monotone in alpha -- 26.36% at 0, 25.75% at 0.25, 26.48% at
    0.5, 31.17% at 1.0 -- i.e. full normalization COSTS 4.8 points of the
    reported metric while a partial weight buys 0.6. Treat alpha ~ 0.25 as the
    starting bracket and select on global val rel-MSE.

    Normalized to mean(w) = 1, which (a) keeps the objective's overall scale --
    and therefore the effective learning rate and the gain/scene gauge -- where
    it was at alpha = 0, and (b) preserves the meaning of --l1-weight and
    --sh-degree-weight, since the prior is added per step UNWEIGHTED and its
    average balance against the data term is unchanged.

    ``max_weight_ratio`` optionally bounds max(w)/min(w), since the weight is
    UNBOUNDED as p_v -> 0 and one anomalously dim view could otherwise take an
    arbitrary share of the objective. It is imposed as a FLOOR ON THE POWER at
    ``p_max / max_weight_ratio**(1/alpha)`` -- any view dimmer than the floor is
    weighted as if it sat there -- NOT as a clamp on the weights themselves.
    That distinction is load-bearing: clamping normalized weights and
    renormalizing is not idempotent, because the renormalization inflates the
    survivors straight back through the cap (this module's gate caught exactly
    that), whereas flooring the power leaves the mean-1 normalization to be
    applied once, afterwards, and exactly.

    It DEFAULTS OFF, so alpha means exactly p^-alpha. The measured B787 training
    spectrum spans 2.02e6 in power, giving a natural weight ratio of 37.7 at
    alpha = 0.25 but 1.4e3 at 0.5 and 5.4e4 at 0.75 -- so any fixed floor tight
    enough to be a real guard would bind at some alphas and not others, silently
    changing what the swept parameter means partway through the sweep. The
    startup line reports the weight range and N_eff, which surfaces a dominating
    view without altering the estimator; set the floor only in response to that.
    A view with zero measured power is always given weight ZERO regardless --
    there is nothing to fit there.

    Multi-GPU: every rank walks the same loader in the same order with no
    randomness, so all ranks derive identical weights with no communication.
    Computed once before training and reused, so it is resume-stable.
    """
    if alpha == 0:
        return None
    powers = []
    for batch in train_loader:
        magnitude_tensor = batch[3]
        powers.append(float((magnitude_tensor.double() ** 2).sum()))
    powers = torch.tensor(powers, dtype=torch.float64)
    live = powers > 0
    if not bool(live.any()):
        raise ValueError("every training viewpoint has zero measured power")

    # log space: p_v runs ~1e-9 in absolute units, and p^-alpha would otherwise
    # be formed from a badly scaled base before the normalization cancels it
    log_p = torch.log(powers[live])
    floored = 0
    if max_weight_ratio and max_weight_ratio > 1 and alpha != 0:
        log_floor = log_p.max() - math.log(max_weight_ratio) / abs(alpha)
        floored = int((log_p < log_floor).sum())
        log_p = log_p.clamp(min=log_floor)
    log_w = -alpha * (log_p - log_p.mean())
    live_w = torch.exp(log_w - log_w.max())
    live_w = live_w / live_w.mean()

    weights = torch.zeros_like(powers)
    weights[live] = live_w

    n_eff_flat = float(powers.sum() ** 2 / (powers ** 2).sum())
    effective = powers * weights
    n_eff_weighted = float(effective.sum() ** 2 / (effective ** 2).sum())
    dead = int((~live).sum())
    rank0_print(
        f"View weighting ON: alpha={alpha}, w_v ~ p_v^-alpha normalized to mean 1 "
        f"(range {weights[live].min():.3g}..{weights[live].max():.3g}); effective view count "
        f"{n_eff_flat:.0f} -> {n_eff_weighted:.0f} of {len(powers)}."
        + (f" {floored} view(s) hit the max_weight_ratio={max_weight_ratio} power floor."
           if floored else "")
        + (f" {dead} zero-power view(s) given weight 0." if dead else "")
    )
    return weights.to(device=device, dtype=torch.float64)


def viewpoint_loss(S_param_pred, frame_data_mag, frame_data_phase, criterion, loss_mode, w_1, w_2,
                   mag_weight=0.0, complex_weight=1.0):
    """Per-viewpoint data-fit loss.

    complex: mean |S_pred - S_meas|^2 over Re/Im -- handles phase wrapping
    natively, weights phase errors by signal magnitude, and (for the explicit
    grid representations) keeps the problem linear least-squares in the
    scene weights. Returns (loss, |dS|^2 sum, |S_meas|^2 sum) so callers can
    also report the global relative MSE sum|dS|^2 / sum|S|^2 (the canonical
    power-weighted metric from the sibling RCS_Comp study).

    mag_weight / complex_weight (SpINRv2's staged supervision, arXiv
    2506.08163v2 Sec. 6.5.5 / Fig. 14): they perturb ground-truth scatterer
    positions and watch each loss term, and find the MAGNITUDE loss has
    strong gradients at ~10 cm scales while the real/imaginary losses only
    become discriminative at ~1 mm -- i.e. a complex loss from scratch is
    poorly conditioned at coarse scales. Their recipe is magnitude-only for
    the first 10% of transmitter positions, then a weighted combination
    (their lambda = 0.5 on Re/Im against 1.0 on magnitude, which is
    mag_weight = 2.0 here). Both weights default to the historical behaviour
    (pure complex), so this is opt-in.

    The returned aux terms stay the DATA-ONLY complex |dS|^2 and |S|^2 no
    matter which weights are set, so the reported rel-MSE remains comparable
    across every run in the project.

    magphase (legacy): w1*MSE(|S|) + w2*MSE(wrapped phase). Kept only for
    back-compat with pre-2026-07-03 runs; its phase term is discontinuous at
    +/-pi and dominated by wrap noise -- do not use for new experiments.
    """
    if loss_mode == "complex":
        S_meas = torch.polar(frame_data_mag, frame_data_phase)         # [Rx, Tx, nf]
        S_pred = S_param_pred.permute(1, 2, 0)                         # [Rx, Tx, nf]
        diff = S_pred - S_meas
        sq_err = diff.real ** 2 + diff.imag ** 2
        loss = complex_weight * sq_err.mean()
        if mag_weight > 0:
            # masked sqrt: |S| is non-differentiable at S = 0 (torch.abs on a
            # complex zero back-propagates NaN), and a bin predicted at exactly
            # zero should contribute zero gradient, not an infinite one
            pow_pred = S_pred.real ** 2 + S_pred.imag ** 2
            mag_pred = torch.where(
                pow_pred > 0, torch.sqrt(pow_pred.clamp_min(1e-30)), torch.zeros_like(pow_pred))
            loss = loss + mag_weight * ((mag_pred - frame_data_mag) ** 2).mean()
        power = (S_meas.real ** 2 + S_meas.imag ** 2).sum()
        return loss, sq_err.sum(), power

    magnitude_cube_pred = torch.abs(S_param_pred).permute(1, 2, 0).contiguous()
    phase_cube_pred = torch.angle(S_param_pred).permute(1, 2, 0).contiguous()
    loss_1 = w_1 * criterion(magnitude_cube_pred, frame_data_mag)
    loss_2 = w_2 * criterion(phase_cube_pred, frame_data_phase)
    return loss_1 + loss_2, loss_1, loss_2


def evaluate(model, val_loader, criterion, device, num_freq_selected, fp_grid, w_1, w_2,
             arr_dist=arr_dist, spacing=spacing, num_rx=num_rx, num_tx=num_tx,
             loss_mode="complex", gain=None, phase_sign=1.0, forward_operator_name="brute",
             compute_dtype=torch.float64, data_format="csv", op_kwargs=None, occlusion=None,
             mag_weight=0.0, return_metrics=False):
    model.eval()
    op_kwargs = op_kwargs or {}
    total_loss = 0.0
    rel_num = 0.0
    rel_den = 0.0

    is_grid_scene = isinstance(model, VoxelGridScene)
    is_sh_grid_scene = isinstance(model, (SHVoxelGridScene, ShardedSHVoxelGridScene))
    is_point_scene = isinstance(model, AdaptivePointSHScene)
    if not is_grid_scene and not is_sh_grid_scene and not is_point_scene:
        x = prepare_model_input(fp_grid.to(device))
        x = positional_encoding(x, pos_encoding_degree, False)

    with torch.no_grad():
        for batch in val_loader:
            if data_format == "npz":
                freqs_tensor, dphi_tensor, dtheta_tensor, magnitude_tensor, phase_tensor, rx_pos_batch, tx_pos_batch = batch
            else:
                freqs_tensor, dphi_tensor, dtheta_tensor, magnitude_tensor, phase_tensor = batch
            magnitude_cube, phase_cube = reshape_measured_cubes(
                magnitude_tensor, phase_tensor, device, num_tx, num_rx
            )

            freqs_tensor = freqs_tensor.squeeze(0).to(device)
            freq_indices = select_freq_indices(freqs_tensor.shape[0], num_freq_selected, device)
            selected_freqs = freqs_tensor[freq_indices]
            frame_data_mag = magnitude_cube[:, :, freq_indices]
            frame_data_phase = phase_cube[:, :, freq_indices]

            k_vector_full = get_kvector(freqs_tensor, cc)
            k_vector = get_kvector(selected_freqs, cc)
            if data_format == "npz":
                rx_pos = rx_pos_batch.squeeze(0).to(device)
                tx_pos = tx_pos_batch.squeeze(0).to(device)
            else:
                rx_pos, tx_pos = get_array_pos(
                    dtheta_tensor.to(device), dphi_tensor.to(device), arr_dist, spacing, num_rx, num_tx, device
                )

            if is_grid_scene:
                scatterer_pos, scatterer_weights = model.active_scatterers()
            elif is_sh_grid_scene or is_point_scene:
                scatterer_pos, scatterer_weights = model.active_scatterers(
                    dtheta_tensor.to(device), dphi_tensor.to(device)
                )
            else:
                w_complex = model(x)
                scatterer_weights = w_complex.view(-1)
                scatterer_pos = fp_grid.to(device).reshape(-1, 3)

            scatterer_weights = apply_occlusion(model, scatterer_weights, rx_pos, tx_pos, occlusion)

            if forward_operator_name == "range":
                S_param_pred = range_forward_operator(
                    freqs_tensor, k_vector_full, rx_pos, tx_pos,
                    scatterer_pos, scatterer_weights,
                    phase_sign=phase_sign, freq_indices=freq_indices, compute_dtype=compute_dtype,
                    **op_kwargs,
                )
            else:
                S_param_pred = forward_operator_lessparallel(
                    selected_freqs, k_vector, rx_pos, tx_pos,
                    scatterer_pos, scatterer_weights,
                    artificial_gain=1.0, p_spectrum=None,
                    omega_scaling="unity", center_freq_hz=None,
                    phase_sign=phase_sign, **op_kwargs,
                )
            # scene-sharded: this rank rendered its own scatterers only
            S_param_pred = all_reduce_sum(S_param_pred)
            if gain is not None:
                S_param_pred = gain(S_param_pred)

            loss, aux_1, aux_2 = viewpoint_loss(
                S_param_pred, frame_data_mag, frame_data_phase, criterion, loss_mode, w_1, w_2,
                mag_weight=mag_weight,
            )
            total_loss += loss.item()
            if loss_mode == "complex":
                rel_num += aux_1.item()
                rel_den += aux_2.item()

    if loss_mode == "complex" and rel_den > 0:
        print(f"Validation Loss: {total_loss:.6e} | Validation relative MSE: {rel_num / rel_den:.4%}")
    else:
        print(f"Validation Loss: {total_loss:.4f}")
    if return_metrics:
        relative_mse = rel_num / rel_den if loss_mode == "complex" and rel_den > 0 else float("nan")
        return {
            "loss": float(total_loss),
            "residual_power": float(rel_num),
            "zero_reference_power": float(rel_den),
            "relative_mse": float(relative_mse),
            "relative_l2": float(math.sqrt(relative_mse)) if math.isfinite(relative_mse) and relative_mse >= 0 else float("nan"),
        }
    return total_loss


def optim_sidecar_path(checkpoint_full_path, rank, world_size):
    """Per-rank AdamW/scheduler state for a scene-sharded run.

    The scene state itself is gathered into the canonical single-GPU layout
    (see ShardedSHVoxelGridScene.full_state_dict) so every eval script and a
    single-GPU resume can read the main checkpoint. The optimizer moments,
    however, are one tensor per SHARD, so they live beside it in a file
    tagged with the world size and are only reused when the world size
    matches; otherwise training resumes with fresh Adam moments (the scene
    itself, which is what the physics depends on, always resumes exactly).
    """
    base = checkpoint_full_path[:-len(".pth.tar")] if checkpoint_full_path.endswith(".pth.tar") \
        else checkpoint_full_path
    return f"{base}.optim_w{world_size}_rank{rank}.pth"


def _atomic_torch_save(state, filepath):
    """Publish a torch checkpoint with an atomic same-directory rename.

    Preemptible jobs must never expose a partially-written recovery file to
    the watcher.  The temporary file lives beside the destination so
    os.replace() stays on one filesystem and is atomic.
    """
    dirname = os.path.dirname(filepath) or "."
    os.makedirs(dirname, exist_ok=True)
    tmp_path = os.path.join(dirname, f".{os.path.basename(filepath)}.tmp.{os.getpid()}")
    try:
        torch.save(state, tmp_path)
        os.replace(tmp_path, filepath)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def load_tensor_checkpoint(checkpoint_path, *, map_location):
    """Load a tensor/primitives-only local checkpoint with restricted pickle.

    New RIFT checkpoints deliberately contain only tensors and standard
    Python containers, including the version-2 RNG payload above.  Be
    explicit about PyTorch's restricted loader so evaluators and recovery do
    not accidentally opt into arbitrary pickle execution.  Very old PyTorch
    releases did not expose ``weights_only``; their historical behavior is
    retained solely for that API-compatibility case.
    """
    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except TypeError as exc:
        # Only the pre-2.0-style API incompatibility may use the historical
        # call shape.  A TypeError raised while interpreting a file must not
        # silently fall through to unrestricted pickle loading.
        message = str(exc)
        if "weights_only" not in message or "unexpected keyword" not in message:
            raise
        return torch.load(checkpoint_path, map_location=map_location)


def validate_npz_sealed_protocol_args(args):
    """Reject contradictory sealed-protocol CLI modes before data loading."""
    if args.npz_sealed_protocol:
        if args.data_format != "npz":
            raise ValueError("--npz-sealed-protocol requires --data-format npz")
        if args.npz_role_manifest is None:
            raise ValueError("--npz-sealed-protocol requires --npz-role-manifest")
        if args.val_from_tail or args.val_cap_axis is not None:
            raise ValueError(
                "--npz-sealed-protocol uses its manifest's explicit roles; do not also pass "
                "--val-from-tail or --val-cap-axis")
    elif args.npz_role_manifest is not None:
        raise ValueError("--npz-role-manifest is valid only with --npz-sealed-protocol")


def preflight_npz_sealed_resume(args):
    """Inspect an NPZ continuation before any eager response-loader can run.

    A legacy NPZ command has historically loaded its entire response archive
    while it builds datasets.  A checkpoint that carries a sealed protocol
    must therefore be recognized *before* that legacy route is reachable.  We
    intentionally inspect only NPZ resumes: CSV commands retain their prior
    checkpoint-loading order, while every possible NPZ response access passes
    through this preflight.  Checkpoints use the existing restricted tensor
    loader and contain no response payload.
    """
    validate_npz_sealed_protocol_args(args)
    if args.data_format != "npz" or args.resume is None:
        return None

    checkpoint = load_tensor_checkpoint(args.resume, map_location="cpu")
    saved_contract = checkpoint.get("sealed_npz_protocol_contract")
    if saved_contract is None:
        if args.npz_sealed_protocol:
            raise ValueError(
                "sealed NPZ resume requires a checkpoint with its complete "
                "sealed_npz_protocol_contract; start a new sealed checkpoint directory")
        return None
    if not isinstance(saved_contract, Mapping):
        raise ValueError("checkpoint sealed_npz_protocol_contract must be an object")
    if not args.npz_sealed_protocol:
        raise ValueError(
            "a checkpoint with sealed_npz_protocol_contract cannot resume through a legacy "
            "or eager NPZ data path; pass --npz-sealed-protocol and its original role manifest")
    return saved_contract


def save_run_checkpoint(checkpoint_full_path, state, model, optimizer, scheduler, atomic=False):
    """Write one training checkpoint, sharded-aware.

    full_state_dict() is a COLLECTIVE (it all-reduces the gathered tensors),
    so every rank must call it; only rank 0 writes the file.
    """
    sharded = isinstance(model, ShardedSHVoxelGridScene)
    state = dict(state)
    # Capture at the actual recovery boundary, after the completed epoch's
    # train+validation frequency draws and before publishing the checkpoint.
    # Each distributed rank executes this function, so rank 0's main file and
    # the per-rank sidecars retain the appropriate process-local state.
    state['rng_state'] = capture_rng_state()
    state['model_state_dict'] = model.full_state_dict() if sharded else model.state_dict()
    state['scheduler_state_dict'] = scheduler.state_dict()
    state['optimizer_state_dict'] = None if sharded else optimizer.state_dict()
    state['world_size'] = get_world_size()
    if get_rank() == 0:
        if atomic:
            _atomic_torch_save(state, checkpoint_full_path)
            print(f"Checkpoint saved atomically to {checkpoint_full_path}")
        else:
            save_checkpoint(state, filepath=checkpoint_full_path)
    if sharded or get_world_size() > 1:
        # Every distributed rank writes a tiny runtime sidecar.  Sharded
        # scenes also need optimizer moments there; replicated scenes need the
        # sidecar only for their rank-local CUDA/Python RNG state.  This is a
        # checkpoint payload, not coordination state.
        os.makedirs(os.path.dirname(checkpoint_full_path) or ".", exist_ok=True)
        sidecar_state = {
            'optimizer_state_dict': optimizer.state_dict() if sharded else None,
            'world_size': get_world_size(),
            'rng_state': state['rng_state'],
        }
        sidecar_path = optim_sidecar_path(checkpoint_full_path, get_rank(), get_world_size())
        if atomic:
            _atomic_torch_save(sidecar_state, sidecar_path)
        else:
            torch.save(sidecar_state, sidecar_path)
    barrier()


def load_run_checkpoint(
    breakpoint_path, model, optimizer, scheduler, gain, device, occlusion=None,
    expected_adaptive_config=None, expected_training_contract=None,
    expected_sealed_npz_protocol_contract=None,
    expected_execution_contract=None,
    require_rng_state=False,
):
    """Resume, sharded-aware. Returns (start_epoch, best_loss)."""
    checkpoint = load_tensor_checkpoint(breakpoint_path, map_location=device)
    validate_sealed_npz_resume_contract(
        checkpoint, expected_sealed_npz_protocol_contract)
    validate_execution_contract(checkpoint, expected_execution_contract)
    if expected_adaptive_config is not None:
        validate_adaptive_resume_config(
            checkpoint, expected_adaptive_config,
            expected_training_contract=expected_training_contract,
            sealed_npz_protocol_contract=expected_sealed_npz_protocol_contract)
    runtime_sidecar = None
    if isinstance(model, ShardedSHVoxelGridScene):
        model.load_full_state_dict(checkpoint['model_state_dict'])
        sidecar = optim_sidecar_path(breakpoint_path, get_rank(), get_world_size())
        if os.path.exists(sidecar):
            runtime_sidecar = load_tensor_checkpoint(sidecar, map_location=device)
            optimizer.load_state_dict(runtime_sidecar['optimizer_state_dict'])
        else:
            print(f"NOTE: no optimizer sidecar for world_size={get_world_size()} at {sidecar} "
                  f"-- resuming the scene exactly but with fresh AdamW moments.")
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        if checkpoint.get('optimizer_state_dict') is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        elif require_rng_state:
            raise ValueError(
                "adaptive-capacity-v2 resume requires optimizer state; a model-only checkpoint "
                "would change subsequent gradients and refinement decisions")
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if gain is not None and checkpoint.get('gain_state_dict') is not None:
        gain.load_state_dict(checkpoint['gain_state_dict'])
    elif gain is not None and require_rng_state:
        raise ValueError(
            "adaptive-capacity-v2 resume requires its GlobalComplexGain state; "
            "a fresh gain would change the coherent trajectory")
    if occlusion is not None and checkpoint.get('occlusion_state_dict') is not None:
        occlusion["scale"].load_state_dict(checkpoint['occlusion_state_dict'])
    if get_world_size() > 1 and runtime_sidecar is None:
        sidecar = optim_sidecar_path(breakpoint_path, get_rank(), get_world_size())
        if os.path.exists(sidecar):
            runtime_sidecar = load_tensor_checkpoint(sidecar, map_location=device)
    rng_payload = (runtime_sidecar or {}).get('rng_state', checkpoint.get('rng_state'))
    restore_rng_state(rng_payload, require_complete=require_rng_state)
    return checkpoint['epoch'], checkpoint['loss']


def _prune_target_for_epoch(epoch, mode, target_active, start_epoch, end_epoch, n_entries):
    """Active-entry count --prune-mode target should aim for at `epoch` (1-based).

    Standard cubic sparsity ramp (Zhu & Gupta 2017): the count falls slowly at
    first and flattens onto the target, rather than taking one huge bite at the
    first check the way the legacy relmax rule did. Past `end_epoch` it returns
    the target exactly, so the schedule has a genuine fixed point -- which is
    the whole reason this mode exists.
    """
    if mode != 'target':
        return None
    if target_active <= 0:
        raise ValueError("--prune-mode target requires --prune-target-active > 0")
    start = max(start_epoch, 1)
    if epoch >= end_epoch or end_epoch <= start:
        return target_active
    frac = (epoch - start) / (end_epoch - start)
    frac = min(max(frac, 0.0), 1.0)
    kept = 1.0 - (1.0 - frac) ** 3          # 0 at start -> 1 at end
    return int(round(n_entries - kept * (n_entries - target_active)))


def train_sar(
    num_epochs, model, train_loader, validation_loader, criterion, optimizer, scheduler,
    device, num_freq_selected, checkpoint_path, w_1, w_2, wandb_run=None, breakpoint_path=None,
    prune_every=0, prune_threshold=0.01, prune_criterion='energy', prune_start_epoch=0,
    prune_mode='relmax', prune_target_active=0, prune_end_epoch=0, prune_min_active=0,
    checkpoint_metric='train',
    grow_every=0, grow_threshold=0.1, grow_threshold_mode='relmax',
    grow_criterion='grad', grow_tail_ratio=0.05,
    split_every=0, split_threshold=0.1, split_max_level=2,
    adaptive_capacity_v2=False, adaptive_refine_every=0,
    adaptive_probe_every=0, adaptive_min_spatial_exposure=1,
    adaptive_min_angular_exposure=1, adaptive_spatial_fraction=0.0,
    adaptive_angular_fraction=0.0, adaptive_spatial_floor=0.0,
    adaptive_angular_floor=0.0, adaptive_cooldown_events=0,
    adaptive_child_maturity_events=1, adaptive_max_active=0,
    l1_weight=0.0, sh_smooth_weight=0.0,
    regularizer_normalization="active_mean", regularizer_reference_active_count=0,
    arr_dist=arr_dist, spacing=spacing, num_rx=num_rx, num_tx=num_tx,
    loss_mode="complex", gain=None, step_every=1, clip_grad_norm=0.0, phase_sign=1.0,
    forward_operator_name="brute", scene_repr=None, compute_dtype=torch.float64,
    data_format="csv", op_kwargs=None, occlusion=None,
    mag_weight=0.0, mag_warmup_epochs=0, view_weight_alpha=0.0, view_weight_max_ratio=0.0,
    val_cap_axis=None, optimizer_requested_recipe=None, sealed_npz_protocol_contract=None,
    execution_contract=None, require_full_resume_state=False,
    adaptive_event_observer=None, engineering_observer=None,
):
    op_kwargs = op_kwargs or {}
    is_grid_scene = isinstance(model, VoxelGridScene)
    is_sh_grid_scene = isinstance(model, (SHVoxelGridScene, ShardedSHVoxelGridScene))
    is_point_scene = isinstance(model, AdaptivePointSHScene)
    if adaptive_capacity_v2:
        if not is_point_scene:
            raise ValueError("--adaptive-capacity-v2 requires --scene-repr point_sh")
        # The CLI constructs both modes correctly, but train_sar is also a
        # public API.  Synchronize once and reject a mislabeled v2 scene
        # rather than silently training without immutable support bounds.
        model.refresh_runtime_flags()
        if not (bool(model.support_bounds_enabled.item())
                and bool(model.compact_sh_eval_enabled.item())):
            raise ValueError(
                "--adaptive-capacity-v2 requires AdaptivePointSHScene with "
                "enforce_support_bounds=True and compact_sh_eval=True")
        if adaptive_refine_every < 1:
            raise ValueError("--adaptive-capacity-v2 requires --adaptive-refine-every >= 1")
        if adaptive_probe_every < 1:
            raise ValueError("--adaptive-capacity-v2 requires --adaptive-probe-every >= 1")
        if split_every or grow_every:
            raise ValueError("--adaptive-capacity-v2 owns refinement scheduling; set --split-every and "
                             "--grow-every to 0")
        if not (0.0 <= adaptive_spatial_fraction <= 1.0
                and 0.0 <= adaptive_angular_fraction <= 1.0):
            raise ValueError("adaptive spatial/angular fractions must be in [0, 1]")
        if adaptive_min_spatial_exposure < 1 or adaptive_min_angular_exposure < 1:
            raise ValueError("adaptive minimum exposures must be >= 1")
        if adaptive_spatial_floor < 0 or adaptive_angular_floor < 0:
            raise ValueError("adaptive score floors must be >= 0")
        if adaptive_cooldown_events < 0:
            raise ValueError("adaptive cooldown events must be >= 0")
        if adaptive_child_maturity_events < 0:
            raise ValueError("adaptive child maturity events must be >= 0")
        if adaptive_max_active < 0:
            raise ValueError("adaptive max active must be >= 0")
        if adaptive_max_active > 0 and int(model.active_mask.sum().item()) > adaptive_max_active:
            raise ValueError(
                "--adaptive-max-active is below the initial active-point count; "
                "choose at least the initial lattice count or prune explicitly before training")
        if ((l1_weight > 0 or sh_smooth_weight > 0)
                and regularizer_normalization != "fixed_initial"):
            raise ValueError(
                "adaptive-capacity-v2 with a nonzero L1/SH prior requires "
                "--regularizer-normalization fixed_initial so zero-weight split siblings "
                "cannot dilute the objective")
    if regularizer_normalization not in {"active_mean", "fixed_initial"}:
        raise ValueError("regularizer_normalization must be 'active_mean' or 'fixed_initial'")
    if regularizer_normalization == "fixed_initial" and regularizer_reference_active_count <= 0:
        if not (is_grid_scene or is_sh_grid_scene or is_point_scene):
            raise ValueError("fixed_initial regularization requires an explicit scene representation")
        regularizer_reference_active_count = all_reduce_int(
            int(model.active_mask.sum().item()), device=model.w_re.device)
    expected_adaptive_config = adaptive_refinement_config(
        enabled=adaptive_capacity_v2,
        refine_every=adaptive_refine_every,
        probe_every=adaptive_probe_every,
        min_spatial_exposure=adaptive_min_spatial_exposure,
        min_angular_exposure=adaptive_min_angular_exposure,
        spatial_fraction=adaptive_spatial_fraction,
        angular_fraction=adaptive_angular_fraction,
        spatial_floor=adaptive_spatial_floor,
        angular_floor=adaptive_angular_floor,
        cooldown_events=adaptive_cooldown_events,
        child_maturity_events=adaptive_child_maturity_events,
        max_active=adaptive_max_active,
        split_max_level=split_max_level,
        regularizer_normalization=regularizer_normalization,
        regularizer_reference_active_count=regularizer_reference_active_count,
    )
    expected_training_contract = None
    if adaptive_capacity_v2:
        expected_training_contract = adaptive_training_contract(
            train_loader=train_loader,
            validation_loader=validation_loader,
            num_freq_selected=num_freq_selected,
            loss_mode=loss_mode, w1=w_1, w2=w_2,
            l1_weight=l1_weight, sh_smooth_weight=sh_smooth_weight,
            regularizer_normalization=regularizer_normalization,
            regularizer_reference_active_count=regularizer_reference_active_count,
            phase_sign=phase_sign, forward_operator_name=forward_operator_name,
            compute_dtype=compute_dtype, data_format=data_format, op_kwargs=op_kwargs,
            arr_dist=arr_dist, spacing=spacing, num_rx=num_rx, num_tx=num_tx,
            prune_every=prune_every, prune_threshold=prune_threshold,
            prune_criterion=prune_criterion, prune_start_epoch=prune_start_epoch,
            prune_mode=prune_mode, prune_target_active=prune_target_active,
            prune_end_epoch=prune_end_epoch, prune_min_active=prune_min_active,
            grow_every=grow_every, grow_threshold=grow_threshold,
            grow_threshold_mode=grow_threshold_mode, grow_criterion=grow_criterion,
            grow_tail_ratio=grow_tail_ratio, split_every=split_every,
            split_threshold=split_threshold, split_max_level=split_max_level,
            step_every=step_every, clip_grad_norm=clip_grad_norm,
            checkpoint_metric=checkpoint_metric, mag_weight=mag_weight,
            mag_warmup_epochs=mag_warmup_epochs, view_weight_alpha=view_weight_alpha,
            view_weight_max_ratio=view_weight_max_ratio, scene_repr=scene_repr,
            gain=gain, occlusion=occlusion, val_cap_axis=val_cap_axis,
            num_epochs=num_epochs, optimizer=optimizer, scheduler=scheduler,
            optimizer_requested_recipe=optimizer_requested_recipe,
        )
    if breakpoint_path:
        print(f"Resuming training from checkpoint: {breakpoint_path}")
        start_epoch, best_loss = load_run_checkpoint(
            breakpoint_path, model, optimizer, scheduler, gain, device, occlusion=occlusion,
            expected_adaptive_config=expected_adaptive_config,
            expected_training_contract=expected_training_contract,
            expected_sealed_npz_protocol_contract=sealed_npz_protocol_contract,
            expected_execution_contract=execution_contract,
            require_rng_state=bool(adaptive_capacity_v2 or require_full_resume_state),
        )
        print(f"Resumed from epoch {start_epoch}, with best loss {best_loss:.4f}")
        if adaptive_capacity_v2:
            # Clear only inactive rows.  The active refinement window is
            # checkpointed and intentionally preserved for exact continuation.
            cleared = model.sanitize_inactive_slots(optimizer=optimizer)
            rank0_print(f"Adaptive v2 resume: sanitized {cleared} inactive slots; "
                        "restored its checkpointed refinement window and RNG state.")
    else:
        start_epoch = 0
        best_loss = float('inf')

    if engineering_observer is not None:
        callback = getattr(engineering_observer, "on_training_start", None)
        if callable(callback):
            callback(
                model=model,
                optimizer=optimizer,
                gain=gain,
                start_epoch=int(start_epoch),
                logical_optimizer_updates=int(start_epoch) * int(math.ceil(len(train_loader) / float(step_every or 1))),
            )

    # This is diagnostic-only and is intentionally opt-in through the narrow
    # adaptive action-gate observer below.  It does not change the ordinary
    # optimizer/scheduler cadence or historical checkpoint payloads.
    updates_per_epoch = (1 if step_every == 0
                         else int(math.ceil(len(train_loader) / float(step_every))))
    logical_optimizer_updates = int(start_epoch) * updates_per_epoch

    # Fixed per-view weights (see compute_view_weights); None at alpha = 0,
    # which leaves every objective in this function byte-identical to the
    # historical one.  Compute after resume restoration so it can never
    # consume or precede the recovery RNG boundary.
    view_weights = compute_view_weights(
        train_loader, view_weight_alpha, device, max_weight_ratio=view_weight_max_ratio)

    training_loss_history = []
    validation_loss_history = []
    # Total entry count the --prune-mode target ramp interpolates DOWN from.
    # The sharded scene holds only its own slice, so take its global n_total.
    n_scene_entries = 0
    if is_grid_scene or is_sh_grid_scene or is_point_scene:
        n_scene_entries = int(getattr(model, 'n_total', model.active_mask.numel()))
    if not is_grid_scene and not is_sh_grid_scene and not is_point_scene:
        # jitter=True preserves this project's existing brute-force training
        # behavior; NUFFT requires jitter=False (see rift/encoding.py) and is
        # wired up separately via --forward-operator. VoxelGridScene/
        # SHVoxelGridScene always use jitter=False (their own fixed regular
        # grid, see sparse_scene.py).
        fp_grid_scene = generate_dynamic_grid(fp_granularity, extent, device, jitter=True)
        x_scene_input = prepare_model_input(fp_grid_scene)
        x_scene_input = positional_encoding(x_scene_input, pos_encoding_degree, False)
    else:
        fp_grid_scene = model.grid_positions

    def params_for_clip():
        return (list(model.parameters())
                + (list(gain.parameters()) if gain is not None else [])
                + (list(occlusion["scale"].parameters()) if occlusion is not None else []))

    def compute_scene(dtheta_dev=None, dphi_dev=None, probe_next_band=False):
        if is_grid_scene:
            return model.active_scatterers()
        if is_sh_grid_scene or is_point_scene:
            if is_point_scene:
                return model.active_scatterers(
                    dtheta_dev, dphi_dev, probe_next_band=probe_next_band)
            return model.active_scatterers(dtheta_dev, dphi_dev)
        w_complex_scene = model(x_scene_input)
        return fp_grid_scene.reshape(-1, 3), w_complex_scene.view(-1)

    def global_grad_norm_and_clip():
        """clip_grad_norm_ over the union of all ranks' shards.

        Scene parameters are DISJOINT across ranks, so their squared norms
        add; the calibration gain is REPLICATED (identical grads on every
        rank) and is counted once, locally, rather than R times. Without
        this the reported grad norm -- and any actual clipping -- would be
        per-shard and would depend on the GPU count.
        """
        scene_sq = torch.zeros((), device=device)
        for p in model.parameters():
            if p.grad is not None:
                scene_sq = scene_sq + p.grad.detach().pow(2).sum()
        total_sq = all_reduce_sum(scene_sq)
        if gain is not None:
            for p in gain.parameters():
                if p.grad is not None:
                    total_sq = total_sq + p.grad.detach().pow(2).sum()
        if occlusion is not None:
            # zeta is replicated like the gain (opacity_volume all-reduces the
            # extinction field first), so it is counted once, not once per rank
            for p in occlusion["scale"].parameters():
                if p.grad is not None:
                    total_sq = total_sq + p.grad.detach().pow(2).sum()
        total_norm = total_sq.sqrt()
        if clip_grad_norm > 0 and float(total_norm) > clip_grad_norm:
            scale = clip_grad_norm / (float(total_norm) + 1e-6)
            for p in params_for_clip():
                if p.grad is not None:
                    p.grad.mul_(scale)
        return total_norm

    def optimizer_step():
        if is_dist():
            grad_norm = global_grad_norm_and_clip()
        else:
            max_norm = clip_grad_norm if clip_grad_norm > 0 else float('inf')
            grad_norm = torch.nn.utils.clip_grad_norm_(params_for_clip(), max_norm=max_norm)
        if is_sh_grid_scene or (is_point_scene and not adaptive_capacity_v2):
            # Must run before optimizer.step()/zero_grad() while .grad still
            # reflects the backward passes since the last step.  Adaptive v2
            # instead records per-view pre-clipping data-fit deltas below.
            model.accumulate_grad_stats()
        optimizer.step()
        optimizer.zero_grad()
        return float(grad_norm)

    def complete_optimizer_step(epoch_number):
        """Run an existing optimizer step and optionally expose completed state.

        The observer is an engineering-only diagnostic seam.  It observes the
        state strictly after ``optimizer.step`` / ``zero_grad`` and cannot
        influence the step, gradient, controller scores, or scheduler.
        """
        nonlocal logical_optimizer_updates
        step_t0 = time.time()
        if engineering_observer is not None:
            callback = getattr(engineering_observer, "before_optimizer_step", None)
            if callable(callback):
                callback(model=model, optimizer=optimizer)
        grad_norm = optimizer_step()
        logical_optimizer_updates += 1
        if adaptive_event_observer is not None:
            callback = getattr(adaptive_event_observer, "on_optimizer_step", None)
            if callable(callback):
                callback(
                    epoch=int(epoch_number),
                    logical_optimizer_updates=int(logical_optimizer_updates),
                    grad_norm=float(grad_norm),
                    seconds=float(time.time() - step_t0),
                )
        if engineering_observer is not None:
            callback = getattr(engineering_observer, "on_optimizer_step", None)
            if callable(callback):
                callback(
                    model=model,
                    optimizer=optimizer,
                    epoch=int(epoch_number),
                    logical_optimizer_updates=int(logical_optimizer_updates),
                    grad_norm=float(grad_norm),
                )
        return grad_norm

    for epoch in range(start_epoch, num_epochs):
        epoch_t0 = time.time()
        model.train()

        # SpINRv2's staged supervision (see viewpoint_loss): magnitude-only
        # while the scene is still far from the truth, because the Re/Im
        # losses are only discriminative at ~1 mm scales, then the full
        # weighted combination. A no-op unless --mag-warmup-epochs is set.
        in_mag_warmup = mag_warmup_epochs > 0 and epoch < mag_warmup_epochs
        complex_weight = 0.0 if in_mag_warmup else 1.0
        if in_mag_warmup and epoch == start_epoch:
            rank0_print(f"Magnitude-only warmup: epochs {epoch}-{mag_warmup_epochs - 1} "
                        f"train on |S| alone (complex term off), then complex + "
                        f"{mag_weight:g}x magnitude.")

        # step_every == 0 (legacy): the scene is rendered once per epoch and
        # every viewpoint backprops through that shared graph, with a single
        # accumulated optimizer step at epoch end. step_every >= 1: the scene
        # must be re-rendered per viewpoint because parameters change between
        # viewpoints. (SH grid scenes are viewpoint-dependent and re-render
        # inside the loop in both modes.)
        scatterer_pos_scene, scatterer_weights_scene = None, None
        if step_every == 0 and not is_sh_grid_scene and not is_point_scene:
            scatterer_pos_scene, scatterer_weights_scene = compute_scene()

        epoch_total_loss = 0.0
        epoch_data_loss_total = 0.0
        epoch_aux1_total = 0.0   # complex: sum |dS|^2   | magphase: weighted mag loss
        epoch_aux2_total = 0.0   # complex: sum |S_meas|^2 | magphase: weighted phase loss
        epoch_reg_total = 0.0
        epoch_l1_reg_total = 0.0
        epoch_sh_degree_reg_total = 0.0
        last_grad_norm = 0.0
        viewpoints_since_step = 0
        # the loaders are batch_size=1 and never shuffled, so the enumeration
        # order is the same every epoch and indexes view_weights consistently
        view_index = 0

        optimizer.zero_grad()

        for batch in train_loader:
            if data_format == "npz":
                freqs_tensor, dphi_tensor, dtheta_tensor, magnitude_tensor, phase_tensor, rx_pos_batch, tx_pos_batch = batch
            else:
                freqs_tensor, dphi_tensor, dtheta_tensor, magnitude_tensor, phase_tensor = batch
            magnitude_cube, phase_cube = reshape_measured_cubes(
                magnitude_tensor, phase_tensor, device, num_tx, num_rx
            )
            freqs_tensor = freqs_tensor.squeeze(0).to(device)

            freq_indices = select_freq_indices(freqs_tensor.shape[0], num_freq_selected, device)
            selected_freqs = freqs_tensor[freq_indices]
            frame_data_mag = magnitude_cube[:, :, freq_indices]
            frame_data_phase = phase_cube[:, :, freq_indices]

            k_vector_full = get_kvector(freqs_tensor, cc)
            k_vector = get_kvector(selected_freqs, cc)
            if data_format == "npz":
                rx_pos = rx_pos_batch.squeeze(0).to(device)
                tx_pos = tx_pos_batch.squeeze(0).to(device)
            else:
                rx_pos, tx_pos = get_array_pos(
                    dtheta_tensor.to(device), dphi_tensor.to(device), arr_dist, spacing, num_rx, num_tx, device
                )

            if is_sh_grid_scene or is_point_scene or step_every > 0:
                scatterer_pos_scene, scatterer_weights_scene = compute_scene(
                    dtheta_tensor.to(device), dphi_tensor.to(device)
                )

            # visibility: view-dependent, so it must be recomputed per
            # viewpoint like the SH evaluation above, and applied to a FRESH
            # weight tensor (never in place -- step_every == 0 reuses one scene
            # graph across the whole epoch)
            weights_view = apply_occlusion(
                model, scatterer_weights_scene, rx_pos, tx_pos, occlusion)

            if forward_operator_name == "range":
                S_param_pred = range_forward_operator(
                    freqs_tensor, k_vector_full, rx_pos, tx_pos,
                    scatterer_pos_scene, weights_view,
                    phase_sign=phase_sign, freq_indices=freq_indices, compute_dtype=compute_dtype,
                    **op_kwargs,
                )
            else:
                S_param_pred = forward_operator_lessparallel(
                    selected_freqs, k_vector, rx_pos, tx_pos,
                    scatterer_pos_scene, weights_view,
                    artificial_gain=1.0, p_spectrum=None,
                    omega_scaling="unity", center_freq_hz=None,
                    phase_sign=phase_sign, **op_kwargs,
                )
            # Scene-sharded multi-GPU: each rank rendered only its own
            # scatterers, and the operator is a sum over scatterers, so the
            # full-scene prediction is the cross-rank sum. Differentiable:
            # backward is the identity (rift/distributed.py), so the scene
            # shards -- disjoint parameters -- need no gradient reduction.
            S_param_pred = all_reduce_sum_grad(S_param_pred)
            if gain is not None:
                # permute to the operator's [nf, Rx, Tx] layout so the
                # warm-start projection compares aligned elements
                gain.maybe_init_scale(
                    S_param_pred, torch.polar(frame_data_mag, frame_data_phase).permute(2, 0, 1)
                )
                S_param_pred = gain(S_param_pred)

            loss_viewpoint, aux_1, aux_2 = viewpoint_loss(
                S_param_pred, frame_data_mag, frame_data_phase, criterion, loss_mode, w_1, w_2,
                mag_weight=mag_weight, complex_weight=complex_weight,
            )
            # reported data loss stays UNWEIGHTED so train rel-MSE and the
            # loss-history CSV remain comparable across every run in the project
            epoch_data_loss_total += loss_viewpoint.item()
            view_slot = view_index
            if view_weights is not None:
                loss_viewpoint = loss_viewpoint * view_weights[view_slot]
            # Keep the actual data-fit objective separate from a prior.  The
            # adaptive-v2 statistic below must not see L1/SH gradients, and it
            # must be captured before optimizer clipping/accumulation changes
            # their scale.  In the legacy path this alias changes nothing.
            data_fit_loss = loss_viewpoint
            view_index += 1
            reg, reg_terms = regularization_loss(
                model,
                l1_weight,
                sh_smooth_weight,
                gain=gain,
                return_terms=True,
                normalization=regularizer_normalization,
                reference_active_count=regularizer_reference_active_count,
            )
            if reg is not None:
                # added per optimizer step, so each step minimizes
                # (viewpoint data fit + prior); the reported rel-MSE (aux
                # terms) stays data-only for comparability across runs
                loss_viewpoint = loss_viewpoint + reg
                epoch_reg_total += reg.item()
                if "l1" in reg_terms:
                    epoch_l1_reg_total += reg_terms["l1"].item()
                if "sh_degree" in reg_terms:
                    epoch_sh_degree_reg_total += reg_terms["sh_degree"].item()

            epoch_total_loss += loss_viewpoint.item()
            epoch_aux1_total += aux_1.item()
            epoch_aux2_total += aux_2.item()

            if adaptive_capacity_v2:
                # ``.grad`` may already contain preceding microbatches and a
                # prior.  Snapshot it, backprop JUST this view's data fit, and
                # take the difference so the refinement window is invariant to
                # optimizer accumulation and excludes regularization.
                def grad_before(parameter):
                    return (torch.zeros_like(parameter) if parameter.grad is None
                            else parameter.grad.detach().clone())

                old_delta_grad = grad_before(model.delta_raw)
                data_fit_loss.backward()
                data_delta_grad = model.delta_raw.grad.detach() - old_delta_grad

                next_re_grad = next_im_grad = None
                # Rotate the sampled view phase across epochs so a stride does
                # not repeatedly probe the same deterministic-loader subset.
                should_probe = ((view_slot + epoch) % adaptive_probe_every == 0)
                if should_probe:
                    probe_pos, probe_weights = compute_scene(
                        dtheta_tensor.to(device), dphi_tensor.to(device), probe_next_band=True)
                    probe_weights = apply_occlusion(model, probe_weights, rx_pos, tx_pos, occlusion)
                    if forward_operator_name == "range":
                        probe_pred = range_forward_operator(
                            freqs_tensor, k_vector_full, rx_pos, tx_pos,
                            probe_pos, probe_weights,
                            phase_sign=phase_sign, freq_indices=freq_indices,
                            compute_dtype=compute_dtype, **op_kwargs,
                        )
                    else:
                        probe_pred = forward_operator_lessparallel(
                            selected_freqs, k_vector, rx_pos, tx_pos,
                            probe_pos, probe_weights,
                            artificial_gain=1.0, p_spectrum=None,
                            omega_scaling="unity", center_freq_hz=None,
                            phase_sign=phase_sign, **op_kwargs,
                        )
                    probe_pred = all_reduce_sum_grad(probe_pred)
                    if gain is not None:
                        # The probe is measurement-only: it shares the current
                        # calibrated gain but never reinitializes or updates it.
                        probe_pred = gain(probe_pred)
                    probe_loss, _, _ = viewpoint_loss(
                        probe_pred, frame_data_mag, frame_data_phase,
                        criterion, loss_mode, w_1, w_2,
                        mag_weight=mag_weight, complex_weight=complex_weight,
                    )
                    if view_weights is not None:
                        probe_loss = probe_loss * view_weights[view_slot]
                    next_re_grad, next_im_grad = torch.autograd.grad(
                        probe_loss, (model.w_re, model.w_im), allow_unused=False,
                    )
                model.accumulate_refinement_data_stats(
                    data_delta_grad, next_re_grad, next_im_grad)
                if reg is not None:
                    # The optimizer still receives exactly data + prior; only
                    # the controller deliberately ignores the latter.
                    reg.backward()
            else:
                # Legacy accumulate-over-the-epoch mode reuses one shared scene
                # graph across viewpoints, so its buffers must be retained.
                loss_viewpoint.backward(retain_graph=(
                    step_every == 0 and not is_sh_grid_scene and not is_point_scene))

            viewpoints_since_step += 1
            if step_every > 0 and viewpoints_since_step >= step_every:
                last_grad_norm = complete_optimizer_step(epoch + 1)
                viewpoints_since_step = 0

        if step_every == 0 or viewpoints_since_step > 0:
            last_grad_norm = complete_optimizer_step(epoch + 1)
        scheduler.step()

        if ((is_grid_scene or is_sh_grid_scene or is_point_scene) and prune_every > 0
                and (epoch + 1) % prune_every == 0 and (epoch + 1) >= prune_start_epoch):
            target_active = _prune_target_for_epoch(
                epoch + 1, prune_mode, prune_target_active, prune_start_epoch,
                prune_end_epoch or num_epochs, n_scene_entries)
            # VoxelGridScene is isotropic (one complex weight, no SH bands), so it has
            # no criterion to choose -- its magnitude test IS the energy test.
            if is_grid_scene:
                n_active, n_total, prune_report = model.prune(
                    threshold_fraction=prune_threshold, mode=prune_mode,
                    target_active=target_active, min_active=prune_min_active)
            elif is_point_scene:
                n_active, n_total, prune_report = model.prune(
                    threshold_fraction=prune_threshold, criterion=prune_criterion,
                    mode=prune_mode, target_active=target_active,
                    min_active=prune_min_active, optimizer=optimizer,
                )
            else:
                n_active, n_total, prune_report = model.prune(
                    threshold_fraction=prune_threshold, criterion=prune_criterion,
                    mode=prune_mode, target_active=target_active,
                    min_active=prune_min_active,
                )
            unit = "points" if is_point_scene else "voxels"
            sched = f', target {target_active}' if prune_mode == 'target' else ''
            rank0_print(f'  - Pruned: {n_active}/{n_total} {unit} active '
                        f'({100 * n_active / n_total:.1f}%{sched})')
            if prune_report:
                rank0_print(f'    {prune_report}')

        if (adaptive_capacity_v2 and is_point_scene
                and (epoch + 1) % adaptive_refine_every == 0):
            snapshot = model.refinement_snapshot(
                max_level=split_max_level,
                min_spatial_exposure=adaptive_min_spatial_exposure,
                min_angular_exposure=adaptive_min_angular_exposure,
                spatial_floor=adaptive_spatial_floor,
                angular_floor=adaptive_angular_floor,
                cooldown_events=adaptive_cooldown_events,
                child_maturity_events=adaptive_child_maturity_events,
            )
            if adaptive_event_observer is not None:
                callback = getattr(adaptive_event_observer, "on_adaptive_event", None)
                if callable(callback):
                    callback(
                        "before",
                        event=int(snapshot["event"]),
                        epoch=int(epoch + 1),
                        snapshot=snapshot,
                        last_grad_norm=float(last_grad_norm),
                        logical_optimizer_updates=int(logical_optimizer_updates),
                    )
            n_split, n_grown, n_active, refine_report = model.apply_refinement_snapshot(
                snapshot,
                spatial_fraction=adaptive_spatial_fraction,
                angular_fraction=adaptive_angular_fraction,
                max_level=split_max_level,
                max_active=adaptive_max_active,
                optimizer=optimizer,
            )
            if adaptive_event_observer is not None:
                callback = getattr(adaptive_event_observer, "on_adaptive_event", None)
                if callable(callback):
                    callback(
                        "after",
                        event=int(snapshot["event"]),
                        epoch=int(epoch + 1),
                        snapshot=snapshot,
                        n_split=int(n_split),
                        n_grown=int(n_grown),
                        n_active=int(n_active),
                        last_grad_norm=float(last_grad_norm),
                        logical_optimizer_updates=int(logical_optimizer_updates),
                    )
            rank0_print(f"  - {refine_report}")

        # Legacy split AFTER prune (a same-epoch prune frees slots the split
        # can use).  v2 takes its joint snapshot above; it never lets split()
        # clear the angular decision window before that decision is made.
        if (is_point_scene and not adaptive_capacity_v2 and split_every > 0
                and (epoch + 1) % split_every == 0):
            n_split, n_active = model.split(
                threshold_fraction=split_threshold, max_level=split_max_level,
                optimizer=optimizer,
            )
            n_free = int((~model.active_mask).sum().item())
            print(f'  - Split: {n_split} parents -> {8 * n_split} children '
                  f'(one inherited heir + {7 * n_split} zero-weight siblings); {n_active} points active, '
                  f'{n_free} spare slots, finest pitch {2 * model.cell_half[model.active_mask].min().item():.4f}m')

        # --grow-criterion angular is criterion-driven, not a ladder: the tail-ratio test is
        # closed-form from the coefficients (no gradient window to fill), so it is CHECKED every
        # epoch by default and each voxel unlocks only when its own spectrum says it is straining
        # at its cap. --grow-every still throttles the CHECK cadence if set.
        _grow_check = grow_every if grow_every > 0 else (1 if grow_criterion == 'angular' else 0)
        if ((is_sh_grid_scene or (is_point_scene and not adaptive_capacity_v2))
                and _grow_check > 0 and (epoch + 1) % _grow_check == 0):
            if grow_criterion == 'angular':
                n_grown, n_active, grow_report = model.grow_angular(tail_ratio_threshold=grow_tail_ratio)
            else:
                n_grown, n_active, grow_report = model.grow(
                    threshold_fraction=grow_threshold, mode=grow_threshold_mode)
            unit = "points" if is_point_scene else "voxels"
            rank0_print(f'  - Grew order: {n_grown}/{n_active} active {unit} (max order now '
                        f'{int(model.order.max().item())})')
            # One run measures the whole threshold->selectivity curve, so tuning
            # --grow-threshold / --grow-tail-ratio does not need a run per value.
            if grow_report:
                rank0_print(f'    {grow_report}')

        num_viewpoints = len(train_loader)
        avg_total_loss = epoch_total_loss / num_viewpoints if num_viewpoints > 0 else 0.0
        avg_data_loss = epoch_data_loss_total / num_viewpoints if num_viewpoints > 0 else 0.0
        training_loss_history.append(avg_total_loss)

        evaluation_result = evaluate(
            model, validation_loader, criterion, device, num_freq_selected, fp_grid_scene, w_1, w_2,
            arr_dist=arr_dist, spacing=spacing, num_rx=num_rx, num_tx=num_tx,
            loss_mode=loss_mode, gain=gain, phase_sign=phase_sign,
            forward_operator_name=forward_operator_name, compute_dtype=compute_dtype,
            data_format=data_format, op_kwargs=op_kwargs, occlusion=occlusion,
            mag_weight=mag_weight, return_metrics=engineering_observer is not None,
        )
        val_metrics = evaluation_result if engineering_observer is not None else None
        val_loss = float(val_metrics["loss"]) if val_metrics is not None else evaluation_result
        validation_loss_history.append(val_loss)

        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else optimizer.param_groups[0]["lr"]
        epoch_elapsed = time.time() - epoch_t0

        print(f'\nEpoch [{epoch+1}/{num_epochs}] Training Summary:')
        wandb_payload = {
            "epoch": epoch + 1,
            "train/loss": avg_total_loss,
            "train/data_loss": avg_data_loss,
            "val/loss": val_loss,
            "lr": current_lr,
            "grad_norm": last_grad_norm,
            "epoch_seconds": epoch_elapsed,
        }
        if loss_mode == "complex":
            train_rel_mse = epoch_aux1_total / epoch_aux2_total if epoch_aux2_total > 0 else float('nan')
            # Preserve the historical line verbatim for remote watcher/log
            # parsers. It has always contained the total training objective
            # (data + priors), despite its legacy label. The next line is the
            # newly explicit data-only value.
            print(f'  - Avg complex-residual MSE (per viewpoint): {avg_total_loss:.6e}')
            print(f'  - Avg complex-residual data MSE (per viewpoint): {avg_data_loss:.6e}')
            print(f'  - Global relative MSE (sum|dS|^2/sum|S|^2): {train_rel_mse:.4%}')
            wandb_payload["train/rel_mse"] = train_rel_mse
            if l1_weight > 0 or sh_smooth_weight > 0:
                avg_reg = epoch_reg_total / num_viewpoints if num_viewpoints > 0 else 0.0
                print(f'  - Avg regularization term (per viewpoint, included in total loss): {avg_reg:.6e}')
                wandb_payload["train/reg"] = avg_reg
                if l1_weight > 0:
                    avg_l1_reg = epoch_l1_reg_total / num_viewpoints if num_viewpoints > 0 else 0.0
                    print(f'    - group-L1 component: {avg_l1_reg:.6e}')
                    wandb_payload["train/reg_l1"] = avg_l1_reg
                if sh_smooth_weight > 0:
                    avg_sh_degree_reg = (
                        epoch_sh_degree_reg_total / num_viewpoints
                        if num_viewpoints > 0 else 0.0
                    )
                    raw_sh_degree_moment = avg_sh_degree_reg / sh_smooth_weight
                    print(
                        f'    - SH degree component: {avg_sh_degree_reg:.6e} '
                        f'(weight {sh_smooth_weight:.3e}; raw moment {raw_sh_degree_moment:.6e})'
                    )
                    wandb_payload["train/reg_sh_degree"] = avg_sh_degree_reg
                    wandb_payload["train/sh_degree_moment"] = raw_sh_degree_moment
        else:
            avg_mag_error = (epoch_aux1_total / w_1) / num_viewpoints if num_viewpoints > 0 else 0.0
            avg_phase_error = (epoch_aux2_total / w_2) / num_viewpoints if num_viewpoints > 0 else 0.0
            print(f'  - Avg Unweighted Magnitude Error (per viewpoint): {avg_mag_error:.6f}')
            print(f'  - Avg Unweighted Phase Error (per viewpoint, rad): {avg_phase_error:.6f}')
            wandb_payload["train/mag_error"] = avg_mag_error
            wandb_payload["train/phase_error"] = avg_phase_error
        if gain is not None:
            g = gain.gain_value()
            print(f'  - Calibration gain: |g| = {abs(g):.4e}, arg(g) = {np.angle(g):.4f} rad')
            wandb_payload["gain_mag"] = abs(g)
            wandb_payload["gain_phase"] = float(np.angle(g))
        if occlusion is not None:
            # zeta is THE number to read on an occlusion arm: it is the one-way
            # optical depth of one average-energy voxel, and zeta -> 0 is
            # exactly the no-occlusion baseline, so a zeta that decays to ~0
            # means the data does not want visibility at this operating point.
            zeta = occlusion["scale"].value
            print(f'  - Occlusion: zeta = {zeta:.4e} '
                  f'(one-way optical depth of an average-energy voxel, key={occlusion["key"]})')
            wandb_payload["occlusion_zeta"] = zeta
        print(f'  - Avg Total Loss (for checkpointing): {avg_total_loss:.6e}')
        print(f'  - Validation Loss: {val_loss:.6e}')
        # %.3e, not %.4f: the losses here run ~1e-9, so the grad norm does too and
        # a fixed-point format prints a flat "0.0000" that reads like a dead
        # gradient when it is nothing of the kind.
        print(f'  - LR: {current_lr:.6e} | GradNorm (last step): {float(last_grad_norm):.3e}')
        print(f'  - Epoch wall-clock (train+val): {epoch_elapsed:.2f}s')

        if wandb_run is not None:
            wandb_run.log(wandb_payload)

        # --checkpoint-metric val selects on HELD-OUT loss. Round 4 (2026-08-02)
        # is why the option exists: the arms that transiently reached val 26.2-
        # 26.6% -- the best numbers this project has produced -- had their train
        # loss still falling at those epochs, so 'best' tracked train and every
        # one of those scenes was overwritten. When the headline metric is
        # novel-view synthesis, checkpoint on the headline metric.
        current_loss_for_checkpointing = (
            val_loss if checkpoint_metric == "val" else avg_total_loss)
        is_new_best = current_loss_for_checkpointing < best_loss
        if is_new_best:
            best_loss = current_loss_for_checkpointing
            print(f"New best loss: {best_loss:.6e}. Saving checkpoint...")

        checkpoint_state = {
            'epoch': epoch + 1,
            'gain_state_dict': gain.state_dict() if gain is not None else None,
            'occlusion_state_dict': (occlusion["scale"].state_dict()
                                     if occlusion is not None else None),
            'occlusion_key': occlusion["key"] if occlusion is not None else None,
            'range_model': op_kwargs.get('range_model', 'sum2'),
            'scene_repr': scene_repr,
            'extent': getattr(model, 'extent', None),
            'granularity': getattr(model, 'granularity', None),
            'l1_weight': float(l1_weight),
            'sh_smooth_weight': float(sh_smooth_weight),
            'sh_degree_weight': float(sh_smooth_weight),
            'sh_degree_penalty': 'laplace_beltrami_l_lplus1',
            'regularizer_normalization': regularizer_normalization,
            'regularizer_reference_active_count': int(regularizer_reference_active_count),
            'view_weight_alpha': float(view_weight_alpha),
            'view_weight_max_ratio': float(view_weight_max_ratio),
            'adaptive_capacity_v2': bool(adaptive_capacity_v2),
            'adaptive_refinement': expected_adaptive_config,
            'adaptive_training_contract': expected_training_contract,
            # Both change what the run MEANS, so they travel with the scene: the
            # eps decides whether the scene was throttled (see --adam-eps), and a
            # non-None cap axis means val is EXTRAPOLATION and its rel-MSE is not
            # comparable to any interpolation-split run.
            'adam_eps': float(optimizer.param_groups[0].get('eps', 1e-8)),
            'val_cap_axis': (None if val_cap_axis is None
                             else [float(a) for a in val_cap_axis]),
            # A recovery checkpoint carries the best-so-far selector value,
            # not the current epoch's value, so best-model selection resumes
            # with exactly the same semantics.
            'loss': best_loss,
        }
        if adaptive_event_observer is not None:
            if epoch == num_epochs - 1:
                callback = getattr(adaptive_event_observer, "finish_training", None)
                if callable(callback):
                    callback()
            state_callback = getattr(adaptive_event_observer, "checkpoint_state", None)
            if callable(state_callback):
                checkpoint_state['adaptive_event_observer_state'] = state_callback()
        # Keep legacy checkpoints byte-for-byte structurally familiar: only
        # an explicit sealed protocol writes this additional continuation
        # contract.  Its reserved-test and unused IDs remain provenance only;
        # neither has a loader or a materialized response tensor here.
        if sealed_npz_protocol_contract is not None:
            checkpoint_state['sealed_npz_protocol_contract'] = sealed_npz_protocol_contract
        if execution_contract is not None:
            checkpoint_state['execution_contract'] = execution_contract

        if is_new_best or epoch == num_epochs - 1:
            checkpoint_suffix = "best" if is_new_best else f"epoch_{epoch+1}"
            if epoch == num_epochs - 1:
                checkpoint_suffix = "final"

            checkpoint_full_path = os.path.join(checkpoint_path, f"checkpoint_{checkpoint_suffix}.pth.tar")
            save_run_checkpoint(
                checkpoint_full_path, checkpoint_state, model, optimizer, scheduler
            )

        # checkpoint_best is a model-selection artifact and may legitimately
        # remain unchanged for many epochs (especially across a pruning loss
        # jump).  Recovery has different semantics: always publish the latest
        # completed epoch so an 8-hour embers chunk loses at most one epoch.
        # checkpoint_final is the recovery artifact for the terminal epoch.
        if epoch != num_epochs - 1:
            latest_path = os.path.join(checkpoint_path, "checkpoint_latest.pth.tar")
            save_run_checkpoint(
                latest_path, checkpoint_state, model, optimizer, scheduler, atomic=True
            )

        if engineering_observer is not None:
            callback = getattr(engineering_observer, "on_epoch_end", None)
            if callable(callback):
                callback(
                    model=model,
                    optimizer=optimizer,
                    epoch=int(epoch + 1),
                    num_epochs=int(num_epochs),
                    val_metrics=val_metrics,
                )

    return training_loss_history, validation_loss_history


def _architecture_defaults(parser, argv):
    """Apply reviewed presets only when the caller selects/defaults an architecture.

    Explicit legacy --scene-repr calls bypass preset injection entirely. Defaults
    come from the reviewed recipe itself, not a second hand-maintained flag list;
    all explicitly supplied user options continue to win through argparse.
    """
    selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    selector.add_argument("--architecture")
    selector.add_argument("--scene-repr")
    selector.add_argument("--object")
    selector.add_argument("--dataset-root")
    selector.add_argument("--npz-path")
    selector.add_argument("--npz-role-manifest")
    selected, _ = selector.parse_known_args(argv)
    architecture = selected.architecture or selected.scene_repr or "adaptive"
    choices = {"adaptive": "point_sh", "point_sh": "point_sh", "grid_sh": "grid_sh", "grid": "grid", "mlp": "mlp"}
    if architecture not in choices:
        parser.error(f"Unknown RIFT architecture: {architecture!r}")
    if selected.architecture and selected.scene_repr and choices[architecture] != selected.scene_repr:
        parser.error("--architecture conflicts with --scene-repr")
    if architecture == "adaptive":
        from rift.b7873200_adaptive_fullscale import fullscale_train_argv
        reviewed = fullscale_train_argv(npz_path="", manifest_path="", checkpoint_root="")
        preset = parser.parse_args(reviewed)
        omitted = {"checkpoint_name", "checkpoint_root", "npz_path", "npz_role_manifest", "execution_contract_label"}
        defaults = {action.dest: getattr(preset, action.dest) for action in parser._actions
                    if action.dest not in omitted and any(flag in reviewed for flag in action.option_strings)}
        parser.set_defaults(**defaults, execution_contract_label="rift_adaptive_v1")
    parser.set_defaults(scene_repr=choices[architecture])
    if selected.object is not None:
        from rift.rift_dataset import DEFAULT_ROOT, resolve_object_inputs
        npz, manifest = resolve_object_inputs(
            object_name=selected.object, dataset_root=selected.dataset_root or DEFAULT_ROOT,
            npz_path=selected.npz_path, role_manifest_path=selected.npz_role_manifest)
        parser.set_defaults(data_format="npz", npz_path=str(npz), npz_role_manifest=str(manifest),
                            npz_sealed_protocol=True, num_train=3200, num_val=1000, num_test=1000,
                            num_tx=16, num_rx=16, num_freq_wanted=600, phase_sign=-1.0,
                            extent=0.15, granularity=48)
    return architecture


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    p.add_argument("--architecture", choices=["adaptive", "grid_sh", "grid", "point_sh", "mlp"], default=None,
                   help="RIFT architecture/preset (default: adaptive). Explicit legacy --scene-repr keeps its old defaults.")
    p.add_argument("--workflow", choices=["train", "adaptive-fullscale"], default="train",
                   help="Optional reviewed adaptive reporting/recovery workflow; use --workflow adaptive-fullscale --help.")
    p.add_argument("--object", default=None, help="One RIFT dataset object or alias (e.g. a320, b787)")
    p.add_argument("--dataset-root", default=None, help="RIFT collection root used with --object")
    p.add_argument("--data-format", choices=["csv", "npz"], default="csv",
                    help="csv (legacy base default): AEDT-style per-viewpoint CSVs in --data-dir, array positions "
                         "derived analytically via get_array_pos(dtheta,dphi). npz (adaptive preset): a single npz file "
                         "(--npz-path) with ground-truth per-viewpoint Tx/Rx element positions "
                         "(rift/npz_dataset.py) -- get_array_pos is not used at all for this format.")
    p.add_argument("--data-dir", default=None, help="Directory of AEDT-simulated viewpoint CSVs (--data-format csv)")
    p.add_argument("--npz-path", default=None, help="Path to a ground-truth-geometry npz file (--data-format npz)")
    p.add_argument("--npz-sealed-protocol", action="store_true",
                   help="Manifest-bound NPZ development roles (on in the adaptive preset; "
                        "explicit opt-in for legacy architectures). Requires "
                        "--npz-role-manifest and constructs only train/validation loaders; reserved-test "
                        "and unused response arrays are never materialized by this training process.")
    p.add_argument("--npz-role-manifest", default=None,
                   help="JSON manifest with explicit ordered train/validation/reserved-test (and optional "
                        "unused) NPZ view IDs. It is valid only with --npz-sealed-protocol.")
    p.add_argument("--checkpoint-name", required=True, help="Subdirectory under checkpoint-root for this run")
    p.add_argument("--checkpoint-root", default="./training_checkpoints")
    p.add_argument(
        "--execution-contract-label",
        default=None,
        help=(
            "Optional stable label for a compact execution contract saved in every checkpoint. "
            "When present, an attempted resume must reproduce the parsed data/scene/physics/fit "
            "configuration exactly.  Unset preserves historical checkpoint behavior."
        ),
    )
    p.add_argument(
        "--require-full-resume-state",
        action="store_true",
        help=(
            "Opt in to exact continuation only: --resume must carry model, AdamW, scheduler, "
            "global-gain, and complete RNG state. Historical commands leave this off and retain "
            "their legacy state-compatible recovery behavior."
        ),
    )
    p.add_argument("--num-train", type=int, default=1000)
    p.add_argument("--num-val", type=int, default=50)
    p.add_argument("--num-test", type=int, default=50)
    p.add_argument("--num-freq-wanted", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--loss", choices=["complex", "magphase"], default="complex",
                    help="complex: mean |S_pred - S_meas|^2 on the complex signal (default; handles "
                         "phase wrapping, weights phase by magnitude). magphase: LEGACY "
                         "w1*MSE(|S|) + w2*MSE(wrapped phase) -- sits at a wrap-noise floor, kept "
                         "only to reproduce pre-2026-07-03 runs.")
    p.add_argument("--step-every", type=int, default=1,
                    help="Optimizer step every N viewpoints (default 1). 0 = LEGACY behavior: "
                         "accumulate gradients over the whole epoch and take a single step. The "
                         "loss being minimized is the same sum over viewpoints either way; this "
                         "only changes the gradient schedule, not the multi-viewpoint (synthetic "
                         "aperture) coherence, which lives in the shared scene parameters.")
    p.add_argument("--clip-grad-norm", type=float, default=0.0,
                    help="Max gradient norm (0 disables clipping, the default). Legacy runs used 1.0, "
                         "which with --step-every 0 limited a whole run to ~epochs*lr total movement.")
    p.add_argument("--no-learn-gain", action="store_true",
                    help="Disable the learnable global complex calibration gain (rift/calibration.py). "
                         "The gain absorbs the forward operator's uncalibrated absolute scale/phase "
                         "vs AEDT port normalization (~550x on the sphere data); it is global, never "
                         "per-viewpoint, so cross-viewpoint coherence is preserved.")
    p.add_argument("--scene-repr", choices=["mlp", "grid", "grid_sh", "point_sh"], default="grid_sh",
                    help="Legacy architecture selector; explicit use preserves old defaults. grid_sh: explicit voxel grid with per-voxel real spherical-"
                         "harmonic coefficients, so reflectivity varies with viewing angle; adaptive "
                         "per-voxel SH order via --grow-every. Chosen as the primary representation "
                         "based on the sibling RCS_Comp study (SH-LS beat 19 neural architectures "
                         "and all localized bases at representing a B787 angular scattering field). "
                         "point_sh: flat continuous-position point list with the same SH angular "
                         "weights. grid: same but one ISOTROPIC complex scalar per voxel. mlp: the "
                         "original implicit representation.")
    p.add_argument("--forward-operator", choices=["brute", "range"], default="brute",
                    help="brute (legacy base default): direct forward_operator_lessparallel. range (adaptive preset): "
                         "range-factorized frequency-axis operator; pass the full frequency grid and "
                         "gather --num-freq-wanted bins internally.")
    p.add_argument("--range-model", choices=["sum2", "product", "none"], default="sum2",
                    help="Geometric spreading in the forward kernel, honored identically by BOTH "
                         "operators since 2026-08-06. sum2 (default, historical): 1/(R_tx+R_rx)^2. "
                         "product: 1/(R_tx*R_rx) -- the physically correct two-way spherical "
                         "spreading and the kernel SpINR/SpINRv2 use (arXiv 2503.23313, 2506.08163v2 "
                         "Sec. 4.2). The two coincide exactly at R_tx = R_rx; at this project's "
                         "operating point (2.85 cm aperture, 10 m standoff) the legs agree to ~0.15%%, "
                         "so the models differ by <0.3%% and the choice is NOT expected to move any "
                         "number on the current npz data -- it becomes material only in genuinely "
                         "near-field / wide-aperture geometries (SpINRv2 sits at 0.23 m standoff; "
                         "AirSAS uses a 0.2 m turntable). Wire it correctly now so the AirSAS port "
                         "is not a scramble. none: unit gain, diagnostics only.")
    p.add_argument("--compute-dtype", choices=["float32", "float64"], default="float64",
                    help="--forward-operator range only: precision for range_forward_operator/"
                         "range_adjoint_operator. float64 (default): matches the validated Stage-B "
                         "exactness gate (global_rel_mse ~1e-20 vs a real target). float32: ~1.7x "
                         "less peak memory (validate_range_operator.py Stage E) but fails the loose "
                         "Stage-B exactness gate by ~4.7x (~4.7e-3 relative error) -- a fp32 "
                         "dynamic-range limit from combining ~100 GHz carrier frequencies with "
                         "~10-100m ranges, not a bug (see EXPERIMENT_MANAGER_HANDOFF.md, "
                         "2026-07-06 reply). Exposed here to compare both empirically rather than "
                         "deciding from the synthetic validation numbers alone.")
    p.add_argument("--point-chunk", type=int, default=262144,
                    help="--forward-operator range only: scatterers rendered per gradient-checkpointed "
                         "chunk. This is the main knob on PEAK ACTIVATION MEMORY: inside a chunk the "
                         "operator materializes ~kernel_width (20) tensors of shape "
                         "[point_chunk, pair_chunk] during the backward recompute. MEASURED cost "
                         "(sum of the chunk graph's saved-tensor storages, 2026-07-26): "
                         "point_chunk * pair_chunk * 672 B at --compute-dtype float64 (416 B at "
                         "float32) -- 2.1x the 20*16B figure this help text used to quote. Even so "
                         "the absolute numbers are small: the 262144 default costs 4.8 GB and "
                         "16384 costs 0.70 GB, so on a >=24 GB card this knob is NOT a binding "
                         "constraint at g96/g128 -- prefer larger chunks (fewer kernel launches) "
                         "and only shrink on an actual OOM. Smaller = less memory, more kernel "
                         "launches, same result (chunking is an exact decomposition of a sum).")
    p.add_argument("--pair-chunk", type=int, default=64,
                    help="--forward-operator range only: Tx*Rx pairs rendered per chunk (256 pairs for "
                         "a 16x16 array). Second memory knob, multiplies with --point-chunk.")
    p.add_argument("--prune-every", type=int, default=0,
                    help="grid/grid_sh/point_sh scene-repr only: prune every N epochs (0 disables pruning)")
    p.add_argument("--prune-threshold", type=float, default=0.01,
                    help="grid/grid_sh/point_sh scene-repr only: what --prune-mode consumes. relmax: "
                         "deactivate entries below this fraction of the max. mass: the fraction of the "
                         "scene's total angular energy this check may discard. target: unused.")
    p.add_argument("--checkpoint-metric", type=str, default="train", choices=["train", "val"],
                    help="which loss 'checkpoint_best' tracks. 'train' (default) preserves the historical "
                         "behavior of every checkpoint on disk. Use 'val' whenever the headline metric is "
                         "novel-view synthesis: in Round 4 the best val epochs (26.2-26.6%%, the best this "
                         "project has produced) were silently overwritten because train loss was still "
                         "falling past them.")
    p.add_argument("--prune-mode", type=str, default="relmax",
                    choices=["relmax", "mass", "target"],
                    help="grid/grid_sh/point_sh scene-repr only: HOW the cut is chosen (--prune-criterion "
                         "is WHAT is measured). 'relmax' (default, legacy) is a RATCHET -- its reference is "
                         "recomputed every check against an ever-more-concentrated distribution, so it has "
                         "no fixed point but the empty scene; it killed every Round 4 arm that used it "
                         "(45%%->24%%->10%%->3%%->...->22 voxels, then the predict-zero floor). Kept only to "
                         "reproduce pre-2026-08-03 runs. Prefer 'mass' (spend a bounded ENERGY budget per "
                         "check) or 'target' (anneal to --prune-target-active and stay there).")
    p.add_argument("--prune-target-active", type=int, default=0,
                    help="--prune-mode target only: the active-entry count to anneal DOWN to, held exactly "
                         "once reached. Set it from the DOF budget the measurement can actually carry, not "
                         "from the grid size (B787 g48: ~296 independent complex DOF, ~4000 counting "
                         "generously, against 110592 voxels).")
    p.add_argument("--prune-end-epoch", type=int, default=0,
                    help="--prune-mode target only: epoch by which --prune-target-active must be reached "
                         "(1-based; 0 = --epochs). The active count follows the standard cubic sparsity "
                         "ramp from --prune-start-epoch to here, gently at first, then holds at the target.")
    p.add_argument("--prune-min-active", type=int, default=0,
                    help="grid/grid_sh/point_sh scene-repr only: floor on the active-entry count, honored "
                         "by EVERY --prune-mode including relmax. Pruning is irreversible, so this is the "
                         "guard rail that makes an over-eager threshold survivable rather than terminal. "
                         "0 disables it.")
    p.add_argument("--prune-criterion", type=str, default="energy", choices=["energy", "dc"],
                    help="grid_sh/point_sh scene-repr only: what pruning measures. 'energy' (default) = total "
                         "angular energy sqrt(sum_lm |c_lm|^2) over unlocked bands, the rotation-invariant "
                         "measure the geometry evals use, so a voxel survives iff it scatters from SOME "
                         "direction. 'dc' = the legacy l=0-only test, which deletes specular scatterers "
                         "(strong l>=1, weak DC) -- kept only to reproduce pre-2026-07-30 runs.")
    p.add_argument("--prune-start-epoch", type=int, default=0,
                    help="grid/grid_sh/point_sh scene-repr only: suppress pruning until this epoch (1-based, "
                         "0 = prune from the first --prune-every hit). Pruning is IRREVERSIBLE, so an "
                         "over-eager early prune is unrecoverable; use this to let the scene settle first.")
    p.add_argument("--sh-max-degree", type=_sh_max_degree_arg, default=10,
                    help="grid_sh/point_sh scene-repr only: highest SH degree ALLOCATED per entry (0-10), or "
                         "'auto' to derive the cap from the PHYSICS -- L = 2k_max * (voxel half-diagonal), "
                         "the angular bandwidth one voxel can carry (rift.spherical_harmonics."
                         "physical_max_degree). Note finer grids get a LOWER cap: a smaller voxel is more "
                         "isotropic. Coefficients up to the cap are always saved in the checkpoint but start "
                         "locked/zeroed unless --sh-init-degree covers them; --grow-every unlocks up to the "
                         "cap, never beyond. 'auto' needs --data-format npz (it reads f_max from the data).")
    p.add_argument("--normalize-scene-scale", action="store_true",
                    help="grid/grid_sh/point_sh only: after init, rescale the scene to rms|w| = 1 over "
                         "active entries and fold the reciprocal into GlobalComplexGain. The (gain, scene) "
                         "scale is an EXACTLY flat direction of the objective, so this changes no loss "
                         "value -- it fixes the gauge, which makes |w| comparable across runs and across "
                         "--granularity (the operator SUMS scatterers, so the natural |w| shrinks as the "
                         "grid densifies) and stops Adam's per-parameter step from depending on where the "
                         "scale happened to drift. Recommended for any density ladder. Skipped on --resume.")
    p.add_argument("--sh-degree-safety", type=float, default=1.0,
                    help="--sh-max-degree auto only: multiplier on the physical bound before ceiling. >1 buys "
                         "slack against it being a band-limit estimate rather than a hard cutoff.")
    p.add_argument("--sh-init-degree", type=int, default=0, choices=list(range(0, 11)),
                    help="grid_sh/point_sh scene-repr only: SH degree every entry STARTS unlocked at (0=isotropic "
                         "start). Must be <= --sh-max-degree")
    p.add_argument("--grow-every", type=int, default=0,
                    help="grid_sh/point_sh scene-repr only: every N epochs, unlock the next SH degree block "
                         "(order += 1, capped at --sh-max-degree) for active entries whose accumulated "
                         "gradient magnitude is large -- the 'grow order instead of splitting' rule. "
                         "0 disables growth (entries stay at --sh-init-degree throughout)")
    p.add_argument("--grow-criterion", type=str, default="grad", choices=["grad", "angular"],
                    help="grid_sh/point_sh scene-repr only: WHAT triggers an order increase. "
                         "'grad' (default, legacy) = accumulated gradient magnitude on a fixed "
                         "--grow-every epoch ladder. 'angular' = Plenoxel-spirit criterion-driven "
                         "growth: unlock the next degree wherever the share of the entry's angular "
                         "energy sitting in its highest unlocked band (the SH argument is the RADAR "
                         "VIEWPOINT direction, so this is the angular derivative w.r.t. radar "
                         "position) is >= --grow-tail-ratio. Checked every epoch unless --grow-every "
                         "throttles it; self-limiting, so entries stop growing once converged.")
    p.add_argument("--grow-tail-ratio", type=float, default=0.05,
                    help="--grow-criterion angular only: unlock the next SH degree when "
                         "sum_{l=order}|c_lm|^2 / sum_{l<=order}|c_lm|^2 >= this. Dimensionless and "
                         "scale-free (the learnable |g| gain and voxel brightness cancel).")
    p.add_argument("--grow-threshold", type=float, default=0.1,
                    help="grid_sh/point_sh scene-repr only, --grow-criterion grad: how much to grow, read "
                         "according to --grow-threshold-mode. relmax: grow entries whose avg gradient "
                         "magnitude since the last grow is >= this fraction of the max among eligible "
                         "(active, not-maxed-out) entries. quantile: grow this FRACTION of eligible "
                         "entries, highest avg gradient first.")
    p.add_argument("--grow-threshold-mode", type=str, default="relmax", choices=["relmax", "quantile"],
                    help="How --grow-threshold is interpreted. 'relmax' (legacy default) is only "
                         "selective when the per-entry gradient has wide dynamic range -- on the B787 "
                         "g48 grid it does NOT (Round 3, 2026-08-01: 0.1 grew 110592/110592 voxels on "
                         "every check, i.e. min avg_grad >= 0.1 * max). 'quantile' asks directly for the "
                         "top q fraction and is selective by construction; prefer it. Every grow check "
                         "prints the measured relmax->fraction-grown curve either way.")
    p.add_argument("--split-every", type=int, default=0,
                    help="point_sh scene-repr only: every N epochs, subdivide high-gradient points' cells "
                         "into 8 half-pitch octants (Gaussian-Splatting-style densification, "
                         "render-preserving -- see AdaptivePointSHScene.split). Requires spare slot "
                         "capacity via --max-points. 0 disables. Do not schedule on the same epochs as "
                         "--grow-every (they consume the same gradient-accumulation window); pairing "
                         "with --prune-every at the same cadence is fine and recommended (prune runs "
                         "first and recycles dead slots).")
    p.add_argument("--split-threshold", type=float, default=0.1,
                    help="point_sh + --split-every only: a point splits if its avg gradient magnitude since "
                         "the last split is >= this fraction of the max among eligible points")
    p.add_argument("--split-max-level", type=int, default=2,
                    help="point_sh + --split-every only: max subdivision depth; each level halves the cell "
                         "pitch (e.g. g24's 12.5cm pitch -> 6.25cm at level 1 -> 3.125cm at level 2)")
    p.add_argument("--adaptive-capacity-v2", action="store_true",
                   help="Opt-in point_sh recipe: take one immutable, pre-clipping data-fit snapshot for "
                        "spatial densification and zero-next-band SH growth. This deliberately does NOT "
                        "change historical --split-every/--grow-every behavior; when enabled, both must be 0.")
    p.add_argument("--adaptive-refine-every", type=int, default=0,
                   help="adaptive-capacity-v2 only: make one joint capacity decision every N epochs after "
                        "aggregating its data-fit evidence; >=1 required with --adaptive-capacity-v2.")
    p.add_argument("--adaptive-probe-every", type=int, default=0,
                   help="adaptive-capacity-v2 only: run one zero-next-band data-fit gradient probe every N "
                        "training views. The deterministic offset rotates by epoch; >=1 required with v2.")
    p.add_argument("--adaptive-min-spatial-exposure", type=int, default=1,
                   help="adaptive-capacity-v2 only: minimum represented training views before a point may split.")
    p.add_argument("--adaptive-min-angular-exposure", type=int, default=1,
                   help="adaptive-capacity-v2 only: minimum next-band probes before a point may unlock SH order.")
    p.add_argument("--adaptive-spatial-fraction", type=float, default=0.0,
                   help="adaptive-capacity-v2 only: bounded top fraction of finite, positive, sufficiently "
                        "exposed spatial scores to densify at each joint event (0 disables spatial action).")
    p.add_argument("--adaptive-angular-fraction", type=float, default=0.0,
                   help="adaptive-capacity-v2 only: bounded top fraction of finite, positive, sufficiently "
                        "probed next-band scores to unlock at each joint event (0 disables angular action).")
    p.add_argument("--adaptive-spatial-floor", type=float, default=0.0,
                   help="adaptive-capacity-v2 only: absolute positive floor on cell-half-width-scaled physical "
                        "position sensitivity; prevents a zero/tied window from consuming spare capacity.")
    p.add_argument("--adaptive-angular-floor", type=float, default=0.0,
                   help="adaptive-capacity-v2 only: absolute positive floor on normalized zero-next-band gradient.")
    p.add_argument("--adaptive-cooldown-events", type=int, default=0,
                   help="adaptive-capacity-v2 only: required intervening refinement events before a selected "
                        "point may be selected again (1 skips the immediately following event).")
    p.add_argument("--adaptive-child-maturity-events", type=int, default=1,
                   help="adaptive-capacity-v2 only: completed refinement events a newly created zero-weight "
                        "sibling must sit out before it can split or unlock another band (default 1).")
    p.add_argument("--adaptive-max-active", type=int, default=0,
                   help="adaptive-capacity-v2 only: active-point ceiling (0 = allocated --max-points capacity).")
    p.add_argument("--max-points", type=int, default=0,
                    help="point_sh scene-repr only: total slot capacity (active + spare) preallocated for "
                         "--split-every densification; tensor and checkpoint shapes stay fixed at this "
                         "size for the whole run. 0 (default): capacity = granularity^3, no spare slots, "
                         "identical to pre-split behavior. Splitting stops (highest-gradient parents win) "
                         "when spare slots run out. Keep <= ~500k: N=1e6 fp64 OOM'd on H200 even with "
                         "gradient checkpointing budgeted (see handoff Stage-E sections).")
    p.add_argument("--pos-lr", type=float, default=3e-3,
                    help="point_sh only: AdamW lr for delta_raw position offsets, with zero weight decay. "
                         "A 3e-3 raw step moves <= cell_half*3e-3, about 0.4mm on the 24^3/extent=3 "
                         "grid, keeping coherent phase steps below roughly lambda/8.")
    p.add_argument("--arr-dist", type=float, default=None,
                    help="Override rift.config.arr_dist (radar standoff, meters) for datasets with a "
                         "different setup, e.g. B787 (50m) vs the default AEDT sphere/cube data (10m)")
    p.add_argument("--num-rx", type=int, default=None,
                    help="Override rift.config.num_rx, e.g. B787 data has 15 Rx elements, not 16")
    p.add_argument("--num-tx", type=int, default=None, help="Physical source Tx count for collection; legacy array count otherwise")
    p.add_argument("--tx-indices", type=int, nargs="+", help="Ordered collection source Tx indices")
    p.add_argument("--rx-indices", type=int, nargs="+", help="Ordered collection source Rx indices")
    p.add_argument("--spacing", type=float, default=None, help="Override rift.config.spacing (array element pitch, meters)")
    p.add_argument("--extent", type=float, default=None,
                    help="Override rift.config.extent (scene box half-width, meters) -- e.g. the "
                         "pec_sphere npz's target is a 1m-radius sphere, much smaller than the default "
                         "AEDT sphere's assumed ~2-3m box.")
    p.add_argument("--granularity", type=int, default=None,
                    help="Override rift.config.fp_granularity (voxel/anchor grid resolution per axis)")
    p.add_argument("--phase-sign", type=float, default=1.0, choices=[-1.0, 1.0],
                    help="Sign of the propagation phase exp(phase_sign*i*k*R) in the forward "
                         "operator -- a PER-SIMULATOR convention, verify with "
                         "scripts/check_phase_sign.py for each new data source. +1 (default): "
                         "confirmed for this project's AEDT/HFSS pipeline (sphere + B787) via "
                         "range-profile causality, and expected for future Ansys AVXcelerate data. "
                         "Beware: naive range fits CANNOT distinguish the signs on this data "
                         "(frequency-grid alias, see forward_operator.py docstring).")
    p.add_argument("--bp-init", type=int, default=16,
                    help="grid/grid_sh/point_sh with --loss complex only: initialize the scene at the scaled "
                         "backprojection (matched-filter) image accumulated over this many training "
                         "viewpoints before the first epoch (0 disables). Starts training at the "
                         "classical SAR image instead of an empty/random scene.")
    p.add_argument("--shell-init-radius", type=float, default=0.0,
                    help="grid/grid_sh/point_sh with --loss complex only: initialize the scene as an ideal "
                         "thin isotropic shell (|w|=1) at this radius (meters) INSTEAD of the "
                         "backprojection image (--bp-init is skipped when this is set). Tests whether "
                         "the BP init's +8-10cm outward shell bias survives when training starts from "
                         "an unbiased shell. 0 (default) disables.")
    p.add_argument("--init-scale", type=float, default=None,
                    help="grid/grid_sh/point_sh scene-repr only: std of the random complex weight init. "
                         "Default: 0.0 with --loss complex (start from the empty scene, so the "
                         "first gradient steps follow the matched-filter/backprojection direction "
                         "instead of first unlearning random noise) and 0.1 with --loss magphase "
                         "(legacy; magphase needs nonzero weights because angle(0) has no gradient).")
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight-decay", type=float, default=0.0,
                    help="AdamW weight decay on the SCENE parameters (default 0: decay biases the "
                         "reflectivity toward an empty scene; legacy runs used 1e-3). The calibration "
                         "gain never receives weight decay.")
    p.add_argument("--l1-weight", type=float, default=0.0,
                    help="grid/grid_sh/point_sh with --loss complex only: weight of a group-L1 sparsity "
                         "prior, |g| * mean over active entries of the L2 norm of each entry's "
                         "(complex, all-SH-coefficient) weight vector, added to every optimizer "
                         "step's loss. The |g| factor (learnable gain magnitude) makes the prior "
                         "scale-invariant -- without it, shrinking w and inflating the gain "
                         "minimizes the penalty without changing predictions. Targets the "
                         "speckle-overfit failure mode (train views fit through diffuse fast-phase "
                         "structure that is uncorrelated with held-out views) and the BP-inherited "
                         "shell smear. 0 (default) disables.")
    p.add_argument("--sh-degree-weight", "--sh-smooth-weight",
                    dest="sh_smooth_weight", metavar="LAMBDA",
                    type=_nonnegative_float_arg, default=0.0,
                    help="grid_sh/point_sh with --loss complex only: weight of the degree-weighted "
                         "angular-energy prior |g|^2 * mean_active sum_lm "
                         "l(l+1)|c_lm|^2. Degree 0 is free; progressively higher SH bands cost "
                         "more, directly taxing view memorization while retaining the full "
                         "degree cap. |g|^2 makes the term invariant to the gain/scene scale "
                         "gauge. --sh-smooth-weight is an exact legacy alias. 0 disables.")
    p.add_argument("--regularizer-normalization", choices=["active_mean", "fixed_initial"],
                   default="active_mean",
                   help="Normalization for --l1-weight and --sh-degree-weight. active_mean is the exact "
                        "historical behavior. fixed_initial divides by the immutable initial active-scene "
                        "count, so an adaptive point split's seven zero-weight children do not dilute the "
                        "heir's prior; required for --adaptive-capacity-v2 when either prior is nonzero.")
    p.add_argument("--occlusion", action="store_true",
                    help="Enable the per-view two-way transmittance factor exp(-2*tau) in the "
                         "forward model (rift/occlusion.py). Until 2026-08-06 the operator had NO "
                         "visibility term at all -- every voxel radiated to every view, including "
                         "voxels behind an opaque target -- while SH-SAS (Eq. 7), DART, RadarSim and "
                         "GeRaF 2.0 all model it; this closes that gap. The factor is frequency- and "
                         "pair-independent (small-aperture argument in the module docstring), so it "
                         "folds into the complex weight and BOTH operators are untouched. "
                         "grid/grid_sh only: the ray march uses nearest-voxel lookup on the lattice, "
                         "so point_sh's continuous positions need a splatting rule that does not "
                         "exist yet. zeta -> 0 recovers this flag being off EXACTLY, so an occlusion "
                         "arm can only match or beat its own baseline on the training objective -- "
                         "read the printed zeta, not just the loss.")
    p.add_argument("--opacity-key", choices=["energy", "dc"], default="energy",
                    help="--occlusion only: what the extinction field is keyed to. energy (default): "
                         "the rotation-invariant total angular energy sqrt(sum_lm |c_lm|^2) over a "
                         "voxel's unlocked bands -- the same measure --prune-criterion energy uses. "
                         "dc: the l=0 term only, i.e. SH-SAS's rho = |sigma_DC|*zeta. dc exists to be "
                         "run as an ARM against energy, not because it is expected to work here: a "
                         "flat conducting plate is perfectly opaque and has almost no isotropic "
                         "return, so DC-keying makes occluders transparent exactly where occlusion "
                         "matters. We already learned this in another guise -- DC-based PRUNING "
                         "deleted the specular scatterers a PEC target is made of.")
    p.add_argument("--occlusion-scale", type=float, default=0.1,
                    help="--occlusion only: initial zeta, the one-way optical depth contributed by "
                         "ONE average-energy voxel. The extinction field is mean-normalized so the "
                         "whole model stays invariant to the (gain, scene) gauge -- without that a "
                         "fixed scale on a raw |w| would mean nothing, since measured |g| spans "
                         "2e-4 to 1e5 across this project's checkpoints. START SMALL: exp(-2*tau) "
                         "saturates, and a saturated scene has no gradient left to walk the opacity "
                         "back down with (at zeta 1.5 on a dense scene everything past the first "
                         "layer sits at T^2 ~ 4e-11). The 0.1 default is ~82%% transparent for an "
                         "average voxel and lets the fit RAISE it; because the field is "
                         "mean-normalized, a shell voxel in a mostly-empty box is well above average "
                         "and goes opaque at a zeta well under 1.")
    p.add_argument("--occlusion-lr", type=float, default=0.0,
                    help="--occlusion only: AdamW lr for log_zeta (0 = use --lr). Its parameter "
                         "group also runs eps=1e-20, because this project's loss is ~1e-9 in "
                         "absolute units and the resulting ~1e-12 gradient sits four decades under "
                         "AdamW's 1e-8 default eps -- with the default, zeta moves ~0.02%% in 4 "
                         "epochs, i.e. not at all. With eps fixed the step is ~lr per VIEWPOINT, so "
                         "zeta can travel lr*num_train in log space per epoch; start at --lr and "
                         "lower it if the printed zeta oscillates.")
    p.add_argument("--occlusion-freeze-scale", action="store_true",
                    help="--occlusion only: hold zeta at --occlusion-scale instead of learning it. "
                         "Use for a controlled sweep of the optical depth; the default (learned) arm "
                         "is the one whose zeta is the reportable measurement.")
    p.add_argument("--occlusion-steps", type=int, default=0,
                    help="--occlusion only: ray-march samples per scatterer. 0 (default) derives it "
                         "from the grid: enough that the WORST-case chord (the box diagonal) is "
                         "sampled at --occlusion-step-frac of a voxel pitch. The step is per-ray "
                         "adaptive, so shorter chords are sampled finer than requested, never coarser.")
    p.add_argument("--occlusion-step-frac", type=float, default=0.5,
                    help="--occlusion only: ray-march step as a fraction of the voxel pitch (default "
                         "0.5). The field is piecewise constant per voxel, so there is nothing to "
                         "resolve below the pitch; 0.5 is the guard against a nearest-voxel march "
                         "skipping a cell.")
    p.add_argument("--occlusion-point-chunk", type=int, default=16384,
                    help="--occlusion only: scatterers per gradient-checkpointed march chunk. Peak "
                         "cost is chunk * steps * 3 floats (fp32); the 16384 default is ~33 MB at "
                         "g48. Independent of --point-chunk, which sizes the operator's chunks.")
    p.add_argument("--view-weight-alpha", type=float, default=0.0,
                   help="Weight each training viewpoint by (its total measured power)^-alpha, "
                        "normalized to mean 1. 0 (default) = the historical unweighted-sum "
                        "objective, exactly. The B787 view sphere is flash-dominated: the "
                        "brightest 20%% of views hold 79.3%% of the power, putting the effective "
                        "training-view count at 286 of 1800 -- below the dataset's own angular "
                        "Nyquist. But the dim 80%% carry 48.9%% of the GLOBAL error mass, so this "
                        "raises headroom in the reported power-weighted metric, not just in "
                        "per-view ones. Non-monotone in alpha (full normalization is worse than "
                        "none on the global metric): sweep it, bracket ~0.25, select on val.")
    p.add_argument("--view-weight-max-ratio", type=float, default=0.0,
                   help="Optional bound on max(w)/min(w), imposed as a floor on per-view power. "
                        "0 (default) = off, so --view-weight-alpha means exactly p^-alpha. The "
                        "B787 training spectrum spans 2.02e6 in power (weight ratio 37.7 at "
                        "alpha=0.25, 1.4e3 at 0.5), so any floor tight enough to guard would bind "
                        "at some alphas and not others and make a sweep non-comparable. Set it "
                        "only if the startup weight range shows one view dominating.")
    p.add_argument("--mag-weight", type=float, default=0.0,
                    help="--loss complex only: weight of an added MSE-on-|S| term, i.e. SpINRv2's "
                         "L = sum||S|-|S~||^2 + lambda*sum(Re^2+Im^2) with lambda=0.5, which is "
                         "--mag-weight 2.0 in this normalization (arXiv 2506.08163v2 Sec. 4.2). "
                         "0 (default) = the historical pure-complex loss. The reported rel-MSE stays "
                         "data-only complex either way, so runs remain comparable.")
    p.add_argument("--mag-warmup-epochs", type=int, default=0,
                    help="--loss complex only: train on |S| ALONE for this many epochs before "
                         "switching on the complex term. SpINRv2's staged supervision (their "
                         "Sec. 6.5.5 / Fig. 14): perturbing ground-truth scatterer positions shows "
                         "the magnitude loss has strong gradients at ~10 cm scales while Re/Im only "
                         "become discriminative at ~1 mm -- a complex loss from scratch is poorly "
                         "conditioned at coarse scales. They use the first 10%% of the run. Requires "
                         "--mag-weight > 0. Relevant here because our sub-bin ambiguity ratio "
                         "(lambda/4 : c/2B = 1:52 at 79 GHz / 3 GHz) is ~19x deeper than the worst "
                         "case they tested.")
    p.add_argument("--w1", type=float, default=1.0, help="Magnitude loss weight (--loss magphase only)")
    p.add_argument("--w2", type=float, default=1000.0, help="Phase loss weight (--loss magphase only)")
    p.add_argument("--t0", type=int, default=10, help="CosineAnnealingWarmRestarts T_0")
    p.add_argument("--t-mult", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-from-tail", action="store_true",
                   help="npz only: reserve the LAST --num-val views (seed-fixed permutation) as a "
                        "FIXED held-out set and grow train from the front, so a --num-train sweep "
                        "scores every level on the same interpolation targets (Nyquist ablation). "
                        "Default off = legacy moving-val behaviour (keeps comparability with prior runs).")
    p.add_argument("--val-cap-axis", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                   help="npz only: hold out the --num-val views closest to this axis as a contiguous "
                        "polar CAP, so validation is angular EXTRAPOLATION (held-out directions lie "
                        "outside the convex hull of the training directions) instead of interpolation. "
                        "Takes precedence over --val-from-tail. Training views are a seeded permutation "
                        "of the non-cap set, never cap-distance ordered. A band-limited angular "
                        "interpolator degrades 28.7%% -> 62.2%% on a power-matched cap; the forward "
                        "operator renders out-of-hull views with the same computation as in-hull ones, "
                        "so this is the split where a scene model should separate from an interpolator.")
    p.add_argument("--adam-eps", type=float, default=1e-8,
                   help="AdamW eps for the SCENE and gain parameter groups. The default 1e-8 is "
                        "PyTorch's and reproduces every run before 2026-08-10 exactly. Adam is "
                        "invariant to gradient rescaling only while |g| stays above eps, and this "
                        "project's per-parameter scene gradients were MEASURED at 5.9e-12 (converged) "
                        "to 1.1e-10 (near-empty scene) -- so the scene has been receiving 0.05%%-1.1%% "
                        "of its nominal lr for the whole of every run, while the gain ran up to 68x "
                        "faster because its own gradient is larger. Set 1e-20 to remove the throttle. "
                        "This CHANGES EVERY TRAJECTORY, so it is a controlled arm, not a silent fix.")
    p.add_argument("--resume", default=None, help="Path to a checkpoint to resume from")
    p.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases")
    p.add_argument("--wandb-project", default="RIFT")
    architecture = _architecture_defaults(p, argv)
    args = p.parse_args(argv)
    args.architecture = "adaptive" if args.scene_repr == "point_sh" and args.adaptive_capacity_v2 else architecture
    if args.object is not None:
        if args.data_format != "npz" or not args.npz_sealed_protocol:
            p.error("--object requires the sealed NPZ data path")
    return args


def main(argv=None, *, adaptive_event_observer_factory=None, engineering_observer_factory=None):
    """Run one RIFT command.

    ``adaptive_event_observer_factory`` and ``engineering_observer_factory`` are
    private opt-in engineering seams for bounded audits. They are absent from
    the CLI and default to ``None``. Explicit legacy representations retain
    their historical recipes; commands without an architecture/representation
    now use the adaptive preset.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    workflow_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    workflow_parser.add_argument("--workflow", choices=["train", "adaptive-fullscale"], default="train")
    workflow, remaining = workflow_parser.parse_known_args(argv)
    if workflow.workflow == "adaptive-fullscale":
        if adaptive_event_observer_factory is not None or engineering_observer_factory is not None:
            raise ValueError("Adaptive full-scale workflow owns its observer; external factories cannot be combined")
        from rift import adaptive_training_workflow
        return adaptive_training_workflow.main(remaining)
    args = parse_args(argv)
    # Inherited presets must not override a sealed selected-acquisition manifest.
    if args.npz_sealed_protocol and args.npz_role_manifest:
        from rift.rift_dataset import collection_manifest
        if collection_manifest(args.npz_role_manifest):
            declared = json.loads(Path(args.npz_role_manifest).read_text()).get('antenna_selection')
            for flag, key in (('--num-tx', 'num_tx'), ('--num-rx', 'num_rx')):
                explicit = any(token == flag or token.startswith(flag+'=') for token in argv)
                indices = getattr(args, key.replace('num_', '')+'_indices')
                if not explicit:
                    if indices is not None:
                        setattr(args, key, len(indices))
                    elif declared:
                        setattr(args, key, declared[key])
    if args.object is not None:
        from rift.rift_dataset import load_object_contract, object_identity
        # Reject swapped named objects before any response access or model work.
        _, named_contract = load_object_contract(args.npz_path, args.npz_role_manifest)
        if named_contract["dataset_identity"] != object_identity(args.object):
            raise ValueError("Selected object does not match its NPZ/manifest")
    # This must stay ahead of both the device setup and the historical eager
    # NPZ branch below.  In particular, a sealed checkpoint may never be
    # silently reopened through the old all-response loader.
    resume_sealed_npz_protocol_contract = preflight_npz_sealed_resume(args)
    # Join the process group BEFORE anything allocates on a GPU, so each rank
    # binds its own device (torchrun's LOCAL_RANK). Single-process runs get
    # (0, 1, "cuda") and nothing else changes.
    rank, world_size, device = init_distributed()
    set_seed(args.seed)
    if world_size > 1:
        print(f"Scene-sharded multi-GPU: {world_size} ranks, one voxel shard each "
              f"(rift/distributed.py). rank {rank} -> {device}")
    else:
        print(f"Using device: {device}")

    sealed_npz_protocol_contract = None
    if args.data_format == "npz":
        if args.npz_path is None:
            raise ValueError("--npz-path is required with --data-format npz")
        if args.npz_sealed_protocol:
            (training_data_loader, validation_data_loader, _,
             sealed_npz_protocol_contract) = build_sealed_npz_dataloaders(
                 args.npz_path,
                 args.npz_role_manifest,
                  num_train=args.num_train,
                  num_val=args.num_val,
                  num_test=args.num_test,
                  resume_sealed_npz_protocol_contract=resume_sealed_npz_protocol_contract,
                  num_tx=args.num_tx, num_rx=args.num_rx,
                  tx_indices=args.tx_indices, rx_indices=args.rx_indices,
              )
            roles = sealed_npz_protocol_contract["role_ids"]
            print(
                "Sealed NPZ development protocol: "
                f"train={len(roles['train'])} validation={len(roles['validation'])} "
                f"reserved_test={len(roles['reserved_test'])} unused={len(roles['unused'])}; "
                "no test DataLoader was constructed."
            )
        else:
            # Keep the historical eager call for explicitly selected legacy
            # architectures; the adaptive preset uses the sealed branch above.
            training_data_loader, validation_data_loader, _ = build_npz_dataloaders(
                args.npz_path, args.num_train, args.num_val, args.num_test, seed=args.seed,
                val_from_tail=args.val_from_tail, cap_axis=args.val_cap_axis
            )
        n_freqs_available = training_data_loader.dataset.freqs.shape[0]
        if args.num_freq_wanted > n_freqs_available:
            print(f"Note: --num-freq-wanted {args.num_freq_wanted} exceeds this npz's {n_freqs_available} "
                  f"frequency bins; every viewpoint will just use all {n_freqs_available}.")
        print(f"Note: --data-format npz uses its OWN phase-sign convention, not necessarily the CSV/AEDT "
              f"default (+1). For data/pec_sphere_fmcw_16t16r_79ghz_r10m_2k.npz this was empirically "
              f"determined to be -1 (see scripts/validate_pec_sphere_coherence.py and project memory); "
              f"currently running with --phase-sign {args.phase_sign}. Verify per npz file, don't assume.")
    else:
        if args.data_dir is None:
            raise ValueError("--data-dir is required with --data-format csv")
        training_data_loader, validation_data_loader, _ = build_dataloaders(
            args.data_dir, args.num_train, args.num_val, args.num_test, device
        )

    resolved_arr_dist = args.arr_dist if args.arr_dist is not None else arr_dist
    if sealed_npz_protocol_contract and sealed_npz_protocol_contract.get('antenna_selection'):
        args.num_tx, args.num_rx = sealed_npz_protocol_contract['response_shape'][1:3]
        args.pair_chunk = min(args.pair_chunk, args.num_tx * args.num_rx)
    resolved_num_rx = args.num_rx if args.num_rx is not None else num_rx
    resolved_num_tx = args.num_tx if args.num_tx is not None else num_tx
    resolved_spacing = args.spacing if args.spacing is not None else spacing
    resolved_extent = args.extent if args.extent is not None else extent
    resolved_granularity = args.granularity if args.granularity is not None else fp_granularity

    resolved_sh_max_degree = args.sh_max_degree
    if args.scene_repr in ("grid_sh", "point_sh"):
        if args.data_format == "npz":
            f_max = float(training_data_loader.dataset.freqs.max())
        else:
            f_max = float(get_freqs().max())
        resolved_sh_max_degree, degree_msg = resolve_sh_max_degree(
            args.sh_max_degree, resolved_extent, resolved_granularity, f_max,
            safety=args.sh_degree_safety,
        )
        rank0_print(f"SH degree cap: {degree_msg}")
        if args.sh_init_degree > resolved_sh_max_degree:
            raise ValueError(
                f"--sh-init-degree {args.sh_init_degree} exceeds the resolved --sh-max-degree "
                f"{resolved_sh_max_degree}. Lower --sh-init-degree (0 is the intended start, with "
                f"--grow-every/--grow-criterion earning higher orders) or raise --sh-max-degree."
            )

    if args.sh_smooth_weight > 0:
        if args.scene_repr not in ("grid_sh", "point_sh"):
            raise SystemExit(
                "--sh-degree-weight/--sh-smooth-weight requires --scene-repr grid_sh or point_sh; "
                f"got {args.scene_repr!r}."
            )
        if args.loss != "complex":
            raise SystemExit(
                "--sh-degree-weight/--sh-smooth-weight is defined for --loss complex only."
            )
        rank0_print(
            "SH degree regularizer ON: "
            f"weight={args.sh_smooth_weight:.3e}, normalization={args.regularizer_normalization}, "
            "penalty=|g|^2 * normalized sum_lm l(l+1)|c_lm|^2 (degree 0 free)."
        )

    if world_size > 1 and args.scene_repr != "grid_sh":
        raise ValueError(
            f"multi-GPU (WORLD_SIZE={world_size}) currently supports --scene-repr grid_sh only "
            f"(got {args.scene_repr}); rift/sharded_scene.py holds the sharded representation. "
            f"Run other representations single-GPU."
        )

    init_scale = args.init_scale if args.init_scale is not None else (0.0 if args.loss == "complex" else 0.1)
    if args.scene_repr == "grid":
        model = VoxelGridScene(resolved_granularity, resolved_extent, device, init_scale=init_scale).to(device)
    elif args.scene_repr == "grid_sh":
        # One voxel shard per rank when distributed; ShardedSHVoxelGridScene is
        # mathematically identical to SHVoxelGridScene (see
        # scripts/validate_distributed_scene.py), so the single-GPU default
        # stays on the original class and prior checkpoints load unchanged.
        scene_cls = ShardedSHVoxelGridScene if world_size > 1 else SHVoxelGridScene
        model = scene_cls(
            resolved_granularity, resolved_extent, device, max_degree=resolved_sh_max_degree,
            init_degree=args.sh_init_degree, init_scale=init_scale,
        ).to(device)
        if world_size > 1:
            n_local = model.w_re.shape[0]
            print(f"  scene: {resolved_granularity}^3 = {resolved_granularity ** 3} voxels x "
                  f"{model.w_re.shape[1]} SH coefficients, ~{n_local} voxels per rank")
    elif args.scene_repr == "point_sh":
        model = AdaptivePointSHScene.from_regular_grid(
            resolved_granularity, resolved_extent, device, max_degree=resolved_sh_max_degree,
            init_degree=args.sh_init_degree, init_scale=init_scale, learn_positions=True,
            capacity=args.max_points if args.max_points > 0 else None,
            enforce_support_bounds=args.adaptive_capacity_v2,
            compact_sh_eval=args.adaptive_capacity_v2,
        ).to(device)
        if args.split_every > 0 and args.max_points <= resolved_granularity ** 3:
            print(f"WARNING: --split-every {args.split_every} but --max-points "
                  f"{args.max_points} leaves no spare slots beyond the initial "
                  f"{resolved_granularity}^3 grid -- splits can only use slots freed by "
                  f"--prune-every. Set --max-points well above granularity^3 for real densification.")
        if args.split_every > 0 and args.grow_every > 0:
            print("WARNING: --split-every and --grow-every are both enabled; they consume the same "
                  "gradient-accumulation window, so on epochs where both fire, grow() sees an empty "
                  "window and no-ops. Prefer enabling only one, or use non-colliding cadences.")
        if args.adaptive_capacity_v2:
            print(
                "Adaptive capacity v2 ON: pre-clipping data-fit statistics, "
                f"joint event every {args.adaptive_refine_every} epoch(s), "
                f"probe stride {args.adaptive_probe_every} views, "
                f"spatial/angular fractions {args.adaptive_spatial_fraction:g}/"
                f"{args.adaptive_angular_fraction:g}, "
                f"minimum exposures {args.adaptive_min_spatial_exposure}/"
                f"{args.adaptive_min_angular_exposure}, child maturity "
                f"{args.adaptive_child_maturity_events} event(s), immutable original support bounds, and "
                "compact active-SH rendering (compute-only; coefficient/Adam allocation stays fixed)."
            )
            if (args.adaptive_spatial_fraction > 0
                    and args.max_points <= resolved_granularity ** 3):
                print("WARNING: adaptive spatial fraction is positive but --max-points has no spare "
                      "slots beyond the initial grid; only angular growth can occur until pruning frees slots.")
    else:
        input_size = pos_encoding_degree * 3 * 2
        model = MLP(input_size, args.hidden_size, output_size=2).to(device)

    if args.loss == "complex" and (args.w1 != 1.0 or args.w2 != 1000.0):
        print("Note: --w1/--w2 only apply to --loss magphase; the complex loss has no "
              "separate magnitude/phase terms to weight. Ignoring them.")

    gain = None if args.no_learn_gain else GlobalComplexGain().to(device)
    compute_dtype = torch.float32 if args.compute_dtype == "float32" else torch.float64
    # chunk sizes are an exact decomposition of the operator's sums, so they
    # trade memory for kernel launches without touching the result.
    # range_model is the one entry both operator families accept, so it rides
    # along in op_kwargs and every call site (train, eval, backprojection init)
    # picks it up without a separate parameter to thread through.
    op_kwargs = {"range_model": args.range_model}
    if args.forward_operator == "range":
        op_kwargs.update(point_chunk=args.point_chunk, pair_chunk=args.pair_chunk)

    occlusion = None
    if args.occlusion:
        if args.scene_repr not in ("grid", "grid_sh"):
            raise SystemExit(
                f"--occlusion needs an explicit voxel lattice to march through; "
                f"--scene-repr {args.scene_repr} is not supported (see rift/occlusion.py).")
        occlusion = {
            "scale": OcclusionScale(
                args.occlusion_scale, learnable=not args.occlusion_freeze_scale).to(device),
            "key": args.opacity_key,
            "n_steps": args.occlusion_steps or None,
            "step_frac": args.occlusion_step_frac,
            "point_chunk": args.occlusion_point_chunk,
        }
        rank0_print(
            f"Occlusion ON: two-way exp(-2*tau), key={args.opacity_key}, "
            f"zeta_0={args.occlusion_scale:g}"
            f"{' (frozen)' if args.occlusion_freeze_scale else ' (learned)'}.")

    if args.mag_warmup_epochs > 0 and args.mag_weight <= 0:
        raise SystemExit("--mag-warmup-epochs needs --mag-weight > 0: with both the complex and "
                         "the magnitude term switched off the warmup loss is identically zero.")
    if (args.mag_weight > 0 or args.mag_warmup_epochs > 0) and args.loss != "complex":
        raise SystemExit("--mag-weight / --mag-warmup-epochs apply to --loss complex only.")

    explicit_scene = args.scene_repr in ("grid", "grid_sh", "point_sh")
    if args.resume:
        if args.shell_init_radius > 0 or args.bp_init > 0:
            print("Resume: skipping shell/backprojection initialization; checkpoint state and RNG are restored "
                  "inside train_sar before any stochastic training work.")
    elif args.shell_init_radius > 0 and explicit_scene and args.loss == "complex":
        if args.bp_init > 0:
            print(f"Note: --shell-init-radius {args.shell_init_radius} overrides --bp-init; "
                  f"skipping the backprojection init.")
        shell_init(model, args.shell_init_radius, resolved_extent, resolved_granularity)
    elif args.shell_init_radius > 0:
        print("Note: --shell-init-radius only applies to grid/grid_sh/point_sh with --loss complex; skipping.")
    elif args.bp_init > 0 and explicit_scene and args.loss == "complex":
        backprojection_init(
            model, training_data_loader, device, args.num_freq_wanted,
            resolved_arr_dist, resolved_spacing, resolved_num_rx, resolved_num_tx,
            max_viewpoints=args.bp_init, phase_sign=args.phase_sign,
            forward_operator_name=args.forward_operator, compute_dtype=compute_dtype,
            data_format=args.data_format, op_kwargs=op_kwargs,
        )
    elif args.bp_init > 0:
        print("Note: --bp-init only applies to grid/grid_sh/point_sh with --loss complex; skipping.")

    scene_scale_a = None
    if args.normalize_scene_scale and explicit_scene and not args.resume:
        scene_scale_a = normalize_scene_scale(model, gain)

    criterion = nn.MSELoss()
    # The gauge rescale w -> a*w divides the scene's effective learning rate by
    # a (AdamW steps by ~lr in absolute units), so the scene lr is multiplied by
    # the same a to leave the trajectory unchanged. Without this the flag is not
    # a reparameterization, it is a silent 1/a learning-rate cut -- which is
    # exactly what killed Round 3. The gain is parameterized as log_mag, so its
    # step is already scale-free and it keeps the unmodified lr.
    scene_lr = args.lr * scene_scale_a if scene_scale_a else args.lr
    if scene_scale_a:
        rank0_print(f"Scene-scale gauge: scene lr {args.lr:.3e} -> {scene_lr:.3e} "
                    f"(x{scene_scale_a:.4e}) so the rescale stays a pure reparameterization.")
    # eps on the SCENE and gain groups. Adam's step is lr*m/(sqrt(v)+eps), so it is
    # scale-free only while |g| >> eps. Measured 2026-08-10 on the converged Round-6
    # solution: per-parameter scene gradients run 5.9e-12, and 1.1e-10 even on a
    # near-empty scene -- 2 to 4 decades UNDER the 1e-8 default. Every run to date has
    # therefore stepped the scene at 0.05%-1.1% of --lr while the gain, whose gradient
    # is larger, ran up to 68x faster in relative terms. --adam-eps 1e-20 removes the
    # throttle; the 1e-8 default reproduces every prior trajectory exactly.
    scene_eps = args.adam_eps
    if isinstance(model, AdaptivePointSHScene):
        param_groups = [
            {"params": [model.w_re, model.w_im], "lr": scene_lr,
             "weight_decay": args.weight_decay, "eps": scene_eps},
            {"params": [model.delta_raw], "lr": args.pos_lr, "weight_decay": 0.0,
             "eps": scene_eps},
        ]
    else:
        param_groups = [{"params": model.parameters(), "lr": scene_lr,
                         "weight_decay": args.weight_decay, "eps": scene_eps}]
    if gain is not None:
        param_groups.append({"params": gain.parameters(), "lr": args.lr,
                             "weight_decay": 0.0, "eps": scene_eps})
    if occlusion is not None and not args.occlusion_freeze_scale:
        # log_zeta is already a scale-free coordinate, so it takes a plain lr and
        # never weight decay (decay on it would pull the scene toward
        # "transparent" for no physical reason).
        #
        # eps: Adam is invariant to rescaling the gradient ONLY while |g| stays
        # above its eps. This project's data-fit loss runs ~1e-9 in absolute
        # units (|S| ~ 1e-5), so d(loss)/d(log_zeta) lands around 1e-12 -- four
        # decades UNDER AdamW's 1e-8 default, which cuts the step by that same
        # factor and freezes the parameter. Measured 2026-08-06: with the
        # default eps, zeta moved 0.02% in 4 epochs. GlobalComplexGain is under
        # the identical throttle and survives only because maybe_init_scale
        # warm-starts it to roughly the right value; zeta has no warm start, so
        # it needs the eps fixed or the experiment measures nothing.
        param_groups.append({"params": occlusion["scale"].parameters(),
                             "lr": args.occlusion_lr or args.lr,
                             "weight_decay": 0.0, "eps": 1e-20})
    optimizer = optim.AdamW(param_groups, lr=args.lr)
    if scene_eps != 1e-8:
        rank0_print(f"AdamW eps on the scene/gain groups: {scene_eps:g} (PyTorch default 1e-8). "
                    f"Measured scene gradients run 5.9e-12 to 1.1e-10, so the default throttles "
                    f"the scene to 0.05%-1.1% of --lr; this arm removes that. NOT comparable to "
                    f"any run before 2026-08-10.")
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.t0, T_mult=args.t_mult, eta_min=1e-6)
    # Record the CLI-level recipe, not the fresh-run-only gauge-scaled scene
    # LR.  On --resume the checkpoint restores the saved effective LR and
    # scheduler state after this strict contract has confirmed the requested
    # knobs are unchanged; it must never re-run scene normalization or BP.
    optimizer_requested_recipe = optimizer_scheduler_requested_recipe(
        optimizer,
        scheduler,
        requested_scene_lr=args.lr if explicit_scene else None,
        normalize_scene_scale=bool(args.normalize_scene_scale and explicit_scene),
    )
    execution_contract = checkpoint_execution_contract(
        args,
        resolved_arr_dist=resolved_arr_dist,
        resolved_spacing=resolved_spacing,
        resolved_num_rx=resolved_num_rx,
        resolved_num_tx=resolved_num_tx,
        resolved_extent=resolved_extent,
        resolved_granularity=resolved_granularity,
        resolved_sh_max_degree=resolved_sh_max_degree,
        init_scale=init_scale,
        optimizer_requested_recipe=optimizer_requested_recipe,
    )

    wandb_run = None
    if args.wandb and rank == 0:
        # every rank logs identical metrics; one run per job, from rank 0
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=args.checkpoint_name, config=vars(args))

    checkpoint_path = os.path.join(args.checkpoint_root, args.checkpoint_name)
    adaptive_event_observer = None
    if adaptive_event_observer_factory is not None:
        if not args.adaptive_capacity_v2:
            raise ValueError(
                "an adaptive event observer requires --adaptive-capacity-v2; "
                "it is not available to historical training modes")
        adaptive_event_observer = adaptive_event_observer_factory(
            args=args,
            model=model,
            optimizer=optimizer,
            gain=gain,
            train_loader=training_data_loader,
            validation_loader=validation_data_loader,
            criterion=criterion,
            device=device,
            num_freq_selected=args.num_freq_wanted,
            phase_sign=args.phase_sign,
            forward_operator_name=args.forward_operator,
            compute_dtype=compute_dtype,
            data_format=args.data_format,
            op_kwargs=op_kwargs,
            occlusion=occlusion,
            sealed_npz_protocol_contract=sealed_npz_protocol_contract,
            checkpoint_path=checkpoint_path,
        )
        if adaptive_event_observer is None:
            raise ValueError("adaptive event observer factory returned None")
    engineering_observer = None
    if engineering_observer_factory is not None:
        engineering_observer = engineering_observer_factory(
            args=args,
            model=model,
            optimizer=optimizer,
            gain=gain,
            train_loader=training_data_loader,
            validation_loader=validation_data_loader,
            criterion=criterion,
            device=device,
            num_freq_selected=args.num_freq_wanted,
            phase_sign=args.phase_sign,
            forward_operator_name=args.forward_operator,
            compute_dtype=compute_dtype,
            data_format=args.data_format,
            op_kwargs=op_kwargs,
            occlusion=occlusion,
            sealed_npz_protocol_contract=sealed_npz_protocol_contract,
            checkpoint_path=checkpoint_path,
            arr_dist=resolved_arr_dist,
            spacing=resolved_spacing,
            num_rx=resolved_num_rx,
            num_tx=resolved_num_tx,
            fp_grid=None,
            w_1=args.w1,
            w_2=args.w2,
        )
        if engineering_observer is None:
            raise ValueError("engineering observer factory returned None")
    training_loss, validation_loss = train_sar(
        args.epochs, model, training_data_loader, validation_data_loader,
        criterion, optimizer, scheduler, device, args.num_freq_wanted,
        checkpoint_path, args.w1, args.w2, wandb_run=wandb_run, breakpoint_path=args.resume,
        prune_every=args.prune_every, prune_threshold=args.prune_threshold,
        prune_criterion=args.prune_criterion, prune_start_epoch=args.prune_start_epoch,
        prune_mode=args.prune_mode, prune_target_active=args.prune_target_active,
        prune_end_epoch=args.prune_end_epoch, prune_min_active=args.prune_min_active,
        checkpoint_metric=args.checkpoint_metric,
        grow_every=args.grow_every, grow_threshold=args.grow_threshold,
        grow_threshold_mode=args.grow_threshold_mode,
        grow_criterion=args.grow_criterion, grow_tail_ratio=args.grow_tail_ratio,
        split_every=args.split_every, split_threshold=args.split_threshold,
        split_max_level=args.split_max_level,
        adaptive_capacity_v2=args.adaptive_capacity_v2,
        adaptive_refine_every=args.adaptive_refine_every,
        adaptive_probe_every=args.adaptive_probe_every,
        adaptive_min_spatial_exposure=args.adaptive_min_spatial_exposure,
        adaptive_min_angular_exposure=args.adaptive_min_angular_exposure,
        adaptive_spatial_fraction=args.adaptive_spatial_fraction,
        adaptive_angular_fraction=args.adaptive_angular_fraction,
        adaptive_spatial_floor=args.adaptive_spatial_floor,
        adaptive_angular_floor=args.adaptive_angular_floor,
        adaptive_cooldown_events=args.adaptive_cooldown_events,
        adaptive_child_maturity_events=args.adaptive_child_maturity_events,
        adaptive_max_active=args.adaptive_max_active,
        l1_weight=args.l1_weight, sh_smooth_weight=args.sh_smooth_weight,
        regularizer_normalization=args.regularizer_normalization,
        arr_dist=resolved_arr_dist, spacing=resolved_spacing, num_rx=resolved_num_rx, num_tx=resolved_num_tx,
        loss_mode=args.loss, gain=gain, step_every=args.step_every, clip_grad_norm=args.clip_grad_norm,
        phase_sign=args.phase_sign, forward_operator_name=args.forward_operator, scene_repr=args.scene_repr,
        compute_dtype=compute_dtype, data_format=args.data_format, op_kwargs=op_kwargs,
        occlusion=occlusion, mag_weight=args.mag_weight, mag_warmup_epochs=args.mag_warmup_epochs,
        val_cap_axis=args.val_cap_axis,
        view_weight_alpha=args.view_weight_alpha,
        view_weight_max_ratio=args.view_weight_max_ratio,
        optimizer_requested_recipe=optimizer_requested_recipe,
        sealed_npz_protocol_contract=sealed_npz_protocol_contract,
        execution_contract=execution_contract,
        require_full_resume_state=bool(args.require_full_resume_state),
        adaptive_event_observer=adaptive_event_observer,
        engineering_observer=engineering_observer,
    )

    if rank == 0:
        loss_path = generate_loss_path(checkpoint_path, "training_validation_losses")
        save_losses(training_loss, validation_loss, loss_path)
        print(f"Saved losses to {loss_path}")

    if wandb_run is not None:
        wandb_run.finish()
    shutdown_distributed()


if __name__ == "__main__":
    raise SystemExit(main())
