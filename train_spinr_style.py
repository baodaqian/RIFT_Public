"""Train the independent SpINR v1 passband adaptation on a sealed RIFT object.

This entry point is intentionally opt-in.  It does not alter the legacy RIFT
trainer or its checkpoints.  It binds the canonical sealed B787 sphere10k
manifest before any response payload is opened, then uses the raw complex
observations in a fixed-scale, signed-real neural-field recipe.

The physics path is deliberately split from the neural path: a no-grad
complex128 range render produces a response cotangent; the range adjoint maps
that cotangent to a real field cotangent; and the FP32 MLP is replayed in
tiles to accumulate exactly one logical four-view update.  This avoids
retaining a renderer graph across all 48^3 quadrature points while preserving
the objective's full-view, all-pair meaning. ``--recipe paper-v1-direct`` uses
direct closed-form selected-bin synthesis and its real-field adjoint, G96/GL2
quadrature and the paper's 1500 epochs. ``paper-v1`` preserves the earlier
FFT-based scene-bin recipe and its shorter budget. The collection default
``budget48-direct`` uses G48 midpoint and a 150-epoch comparison budget/schedule;
``budget48-direct-1500`` preserves its earlier 1500-epoch identity. The bare CLI default
``legacy-midpoint`` retains the historical G48/all-bin recipe for old callers.

This file implements a development/pilot entry point only.  It performs no
Slurm action; cluster validation and any production request remain manager
owned.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import signal
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from rift.config import cc
from rift.forward_operator import get_kvector
from rift.range_operator import range_adjoint_operator, range_forward_operator
from rift.spinr_style import (
    SPINR_STYLE_HIDDEN_LAYERS,
    SPINR_STYLE_HIDDEN_WIDTH,
    SPINR_STYLE_INPUT_FEATURES,
    SPINR_STYLE_METHOD_ID,
    SPINR_STYLE_PARAMETER_COUNT,
    SPINR_STYLE_PHASE_SIGN,
    SPINR_STYLE_RANGE_MODEL,
    SPINR_STYLE_RECIPE_ID,
    SPINR_STYLE_SUPPORT_M,
    SealedRawComplexViews,
    build_spinr_style_acquisition_identity,
    SpinrStyleINR,
    build_sealed_raw_complex_views,
    midpoint_grid,
    gauss_legendre_cell_grid,
    scale_field_to_renderer_weights,
    spinr_style_objective,
    validate_spinr_style_acquisition_identity,
)
from rift.spinr_fidelity import (
    PAPER_RECIPE, DIRECT_RECIPE, BUDGET48_RECIPE, BUDGET150_RECIPE, PAPER_EPOCHS,
    COMPARISON_EPOCHS, PAPER_REFERENCE, PARENT_GRID, NODES_PER_CELL,
    scene_range_bin_mask, spectral_partition_terms,
)
from rift.spinr_direct import render_selected_bins, selected_bin_field_vjp, selected_bin_objective
from train import (
    _load_sealed_npz_protocol_contract,
    _validate_saved_sealed_npz_protocol_contract,
    capture_rng_state,
    load_tensor_checkpoint,
    restore_rng_state,
)

try:  # ``resource`` is available on PACE/Linux but not standard Windows Python.
    import resource
except ImportError:  # pragma: no cover - platform-specific compatibility path
    resource = None  # type: ignore[assignment]


GOTCHA_BACKEND = {
    "schema": "rift_gotcha_backend_v1",
    "method": "spinr",
    "callable": "run_gotcha",
    "selection_unit": "pass_sector",
    "joint_passes": True,
    "native_frequency_policy": "ragged_exact",
    "polarizations": ["hh", "hv", "vh", "vv"],
    "metric_domain": "roi_projected_native_complex_with_full_native_diagnostic",
    "default_config": {"epochs": 150, "grid_size": 48, "nodes_per_cell": 1,
                       "pulse_batch_size": 1024, "seed": 42, "neural_point_tile": 4096,
                       "renderer_point_tile": 512, "checkpoint_every": 10, "validation_every": 5},
    "numerical_convergence": "unvalidated_on_native_fitted_fields",
}


def run_gotcha(*, dataset, output_dir, config, device, resume):
    """Native root-dispatch socket; synthetic-data recipes remain unchanged."""
    from rift.spinr_gotcha_training import run
    return run(dataset, output_dir, config, device=device, resume=resume)


DEFAULT_B787_NPZ_PATH = "/storage/home/hcoda1/1/dbao31/r-jromberg3-0/RIFT/data/b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz"
DEFAULT_B787_MANIFEST_PATH = (
    "/storage/scratch1/1/dbao31/rift_round8b_impl_20260810/splits/round8b/"
    "b78710k_interp_seed42_train3200_val1000_test1000_v1.json"
)

CANONICAL_TRAIN_COUNT = 3200
CANONICAL_VALIDATION_COUNT = 1000
CANONICAL_TEST_COUNT = 1000
CANONICAL_UNUSED_COUNT = 4800
CANONICAL_UPDATES_PER_EPOCH = 800
CANONICAL_VIEW_BATCH = 4
CANONICAL_SEED = 42
CANONICAL_MAX_EPOCHS = 300
CANONICAL_MIN_EPOCHS = 150
CANONICAL_VALIDATION_EVERY = 5
CANONICAL_TRAIN_DIAGNOSTIC_COUNT = 128
CANONICAL_INIT_SCALE_COUNT = 32

# No earlier SpINR-style run was released to PACE.  Version two makes the
# stricter pre-payload model/optimizer validation and clipping accumulator an
# honest new checkpoint contract instead of falsely accepting a v1 artifact.
# Legacy RIFT checkpoint formats and renderer defaults are untouched.
CHECKPOINT_FORMAT = "rift_spinr_style_b787_v2"
EXECUTION_SCHEMA = "rift_spinr_style_execution_v2"
_TERMINAL_STOP_REASONS = frozenset({
    "plateau_three_consecutive_10_epoch_windows",
    "development_epoch_budget_reached",
    "epoch_budget_reached",
})
_CHECKPOINT_FILENAMES = frozenset({
    "checkpoint_latest.pth.tar",
    "checkpoint_best.pth.tar",
    "checkpoint_epoch_150.pth.tar",
    "checkpoint_final.pth.tar",
    "metrics_history.json",
})


def _is_managed_checkpoint_temp(path: Path) -> bool:
    """Recognize only this entry point's interrupted atomic-save temporaries."""

    if not path.is_file():
        return False
    for filename in _CHECKPOINT_FILENAMES:
        prefix = f".{filename}.tmp."
        if path.name.startswith(prefix) and path.name[len(prefix):].isdigit():
            return True
    return False


def _finite_positive(value: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive, got {value!r}")
    return value


def _set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _disable_tf32() -> None:
    """Make the initial numerical path explicitly FP32/FP64, not TF32."""

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def _peak_host_rss_bytes() -> int | None:
    """Best available process RSS high-water mark without a new dependency."""

    if resource is None:
        return None
    try:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (AttributeError, OSError):
        return None
    # Linux reports KiB; macOS reports bytes.  PACE is Linux, but retain a
    # conservative cross-platform interpretation for local syntax fixtures.
    return value * 1024 if os.name != "darwin" else value


def _enforce_memory_gates(*, device: torch.device, host_rss_limit_gib: float) -> dict[str, float]:
    """Fail loudly when the declared allocated-node memory envelope is exceeded."""

    limit_bytes = _finite_positive(host_rss_limit_gib, "host RSS limit GiB") * (1024.0 ** 3)
    report: dict[str, float] = {"host_rss_limit_gib": float(host_rss_limit_gib)}
    rss = _peak_host_rss_bytes()
    if rss is not None:
        report["peak_host_rss_gib"] = rss / (1024.0 ** 3)
        if rss > 0.80 * limit_bytes:
            raise MemoryError(
                f"peak host RSS {report['peak_host_rss_gib']:.2f} GiB exceeds 80% of the "
                f"declared {host_rss_limit_gib:.2f} GiB allocation")
    if device.type == "cuda":
        allocated = int(torch.cuda.max_memory_allocated(device))
        total = int(torch.cuda.get_device_properties(device).total_memory)
        report["peak_gpu_allocated_gib"] = allocated / (1024.0 ** 3)
        report["gpu_total_gib"] = total / (1024.0 ** 3)
        if allocated > 0.80 * total:
            raise MemoryError(
                f"peak GPU allocation {report['peak_gpu_allocated_gib']:.2f} GiB exceeds 80% of "
                f"device capacity {report['gpu_total_gib']:.2f} GiB")
    return report


def _atomic_torch_save(state: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        torch.save(dict(state), temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json_save(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


PAPER_BATCH_TEXT = "1024 measurement channels per update"
LEGACY_BATCH_SETTING = "four whole views group 1024 channels; author grouping unknown"


def _recipe_identity(recipe: str = "legacy-midpoint", num_train=None, antenna_selection=None) -> dict[str, Any]:
    """Scientific settings that a resume must not change, with the actual update grouping."""

    return _selected_batch_metadata(_declared_recipe_identity(recipe, num_train, antenna_selection))


def _selected_batch_metadata(identity: dict[str, Any]) -> dict[str, Any]:
    """State the measurement channels one update groups: four views times the selected pairs.

    The paper specifies 1024 per batch. That holds for the historical 16 x 16
    acquisition (4 x 256); with fewer selected pairs it is not honored, so it
    leaves ``specified`` like the 1500-epoch item does.
    """
    fidelity = identity.get("fidelity")
    pairs = identity["batching"]["all_pairs"]
    if fidelity is None or CANONICAL_VIEW_BATCH * pairs == 1024:
        return identity
    fidelity["specified"].remove(PAPER_BATCH_TEXT)
    settings = fidelity["benchmark_settings"]
    settings[settings.index(LEGACY_BATCH_SETTING)] = (
        f"four whole views per update = {CANONICAL_VIEW_BATCH * pairs} measurement channels "
        f"({pairs} selected pairs each); paper specifies 1024; author grouping unknown")
    return identity


def _legacy_batch_metadata(identity: Mapping[str, Any]) -> Mapping[str, Any]:
    """The text checkpoints written before 2026-09-22 carry: '1024 channels' for every acquisition."""
    fidelity = identity.get("fidelity")
    if fidelity is None or PAPER_BATCH_TEXT in fidelity["specified"]:
        return identity
    legacy = copy.deepcopy(dict(identity))
    legacy["fidelity"]["specified"].append(PAPER_BATCH_TEXT)
    settings = legacy["fidelity"]["benchmark_settings"]
    settings[next(i for i, text in enumerate(settings) if text.startswith("four whole views per update ="))] = LEGACY_BATCH_SETTING
    return legacy


def _declared_recipe_identity(recipe: str = "legacy-midpoint", num_train=None, antenna_selection=None) -> dict[str, Any]:
    """Scientific settings that a resume must not change.

    Memory tiles are deliberately absent: they are operational fallbacks and
    may change without changing the four-view objective or the recipe's data.
    """

    from rift.rift_dataset import training_count
    num_train = training_count(CANONICAL_TRAIN_COUNT if num_train is None else num_train)
    if num_train < CANONICAL_INIT_SCALE_COUNT or num_train % CANONICAL_VIEW_BATCH:
        raise ValueError("SpINR requires at least 32 training views in complete four-view batches")
    identity = {
        "method_id": SPINR_STYLE_METHOD_ID,
        "recipe_id": SPINR_STYLE_RECIPE_ID,
        "network": {
            "support_m": SPINR_STYLE_SUPPORT_M,
            "hidden_layers": 6,
            "hidden_width": SPINR_STYLE_HIDDEN_WIDTH,
            "parameter_count": SPINR_STYLE_PARAMETER_COUNT,
            "signed_real_output": True,
            "extra_trainable_gain": False,
            "network_dtype": "float32",
        },
        "operator": {
            # G=48 is the sole trainable quadrature.  G=96/G=192 comparisons
            # belong to a separate evaluator and cannot be smuggled into a
            # continuation through this entry point.
            "grid_size": 48,
            "phase_sign": SPINR_STYLE_PHASE_SIGN,
            "range_model": SPINR_STYLE_RANGE_MODEL,
            "physics_dtype": "float64_complex128",
            "frequency_transform": "fft_norm_forward_all_600",
            "pairs": "all_16x16",
        },
        "normalization": {
            "training_signal_energy": f"raw_mean_abs_squared_all_{num_train}_train",
            "initial_output_scale": "fixed_0.1_sqrt_obs_over_random_first32_train",
            "learned_gain": False,
        },
        "loss": {
            "range_magnitude_squared_weight": 1.0,
            "range_complex_squared_weight": 0.5,
            "global_multiplier": "num_frequency_over_training_raw_mean_power",
        },
        "optimizer": {
            "name": "Adam",
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.0,
            "clip_global_l2": 1.0,
            "cosine_final_lr": 1e-5,
            "cosine_max_epochs": CANONICAL_MAX_EPOCHS,
            "restarts": False,
        },
        "batching": {
            "logical_view_batch": CANONICAL_VIEW_BATCH,
            "updates_per_epoch": num_train // CANONICAL_VIEW_BATCH,
            "all_frequencies": 600,
            "all_pairs": 256,
            "epoch_seed": CANONICAL_SEED,
        },
        # These decisions do not alter an individual renderer evaluation, but
        # they do define when validation, checkpoint selection, and a terminal
        # plateau decision occur.  A continuation must not silently inherit a
        # differently scheduled state machine from a later source revision.
        "monitoring": {
            "validation_every_epochs": CANONICAL_VALIDATION_EVERY,
            "train_diagnostic_views": min(CANONICAL_TRAIN_DIAGNOSTIC_COUNT, num_train),
            "matched_exposure_epoch": CANONICAL_MIN_EPOCHS,
            "plateau": "three_consecutive_10_epoch_running_best_below_1pct",
            "selection": "lowest_full_validation_coherent_relative_mse_earliest_tie",
        },
    }
    if antenna_selection:
        from rift.antenna_selection import validate_selection
        acquisition = validate_selection(antenna_selection)
        identity['antenna_selection'] = acquisition
        identity['operator']['pairs'] = 'all_selected_source_pairs'
        identity['batching']['all_pairs'] = acquisition['num_tx'] * acquisition['num_rx']
    if recipe == "legacy-midpoint":
        return identity
    if recipe not in ("paper-v1", "paper-v1-direct", "budget48-direct", "budget48-direct-1500"):
        raise ValueError(f"Unknown SpINR recipe {recipe!r}")
    identity["recipe_id"] = PAPER_RECIPE
    identity["reference"] = {
        "paper": PAPER_REFERENCE,
        "implementation": "independent; no official executable parity established",
        "acquisition_adaptation": "native passband frequencies and exact bistatic geometry",
        "local_choices": "MLP/encoding/initialization, quadrature, conditioning, optimizer and budget",
    }
    identity["operator"].update({
        "grid_size": PARENT_GRID,
        "quadrature": "tensor_gauss_legendre_per_cell",
        "nodes_per_cell": NODES_PER_CELL,
        "integration_points": (PARENT_GRID*NODES_PER_CELL)**3,
        "frequency_transform": "fft_norm_forward_scene_bins",
        "scene_bins": "per_pair_cube_distance_bounds_floor_ceil_modulo_n_v1",
        "native_frequencies_retained": True,
    })
    identity["loss"]["bin_reduction"] = "sum_selected_bins_over_pairs_and_train_raw_power"
    identity["readout"] = {"quantity": "abs(initial_output_scale*sigma)",
                           "normalization": "divide_by_max", "threshold": "explicit_fixed"}
    if recipe in ("paper-v1-direct", "budget48-direct", "budget48-direct-1500"):
        identity["recipe_id"] = DIRECT_RECIPE
        identity["operator"]["frequency_transform"] = "closed_form_selected_finite_dft_bins"
        identity["operator"]["neural_vjp"] = "selected_bin_real_field_adjoint_then_tiled_neural_replay"
        identity["optimizer"]["cosine_max_epochs"] = PAPER_EPOCHS
        identity["monitoring"].pop("matched_exposure_epoch")
        identity["monitoring"]["diagnostic_milestone_epoch"] = CANONICAL_MIN_EPOCHS
        identity["monitoring"]["plateau"] = "disabled_fixed_paper_epoch_budget"
        identity["monitoring"]["paper_epochs"] = PAPER_EPOCHS
        identity["monitoring"]["shorter_explicit_runs"] = "development_only"
        identity["reference"]["local_choices"] = (
            "undisclosed MLP/encoding/initialization, quadrature, conditioning, optimizer, "
            "batch ordering and readout extraction; fixed before benchmark outcomes")
        identity["fidelity"] = {
            "specified": ["real position-only field", "product spreading", "coherent integral",
                          "direct selected DFT bins", "magnitude squared plus 0.5 complex squared",
                          "1500 epochs", PAPER_BATCH_TEXT],
            "undisclosed": ["MLP depth/width/activation/encoding", "weight initialization",
                            "optimizer/lr schedule/clipping", "signal normalization/output scale",
                            "integration rule", "measurement ordering", "geometry extraction"],
            "benchmark_settings": ["native carrier/frequency grid/phase convention",
                                   "exact source Tx/Rx and spherical poses instead of cylindrical monostatic data",
                                   f"object-bound support and sealed {num_train}/1000 split",
                                   LEGACY_BATCH_SETTING,
                                   "validation selection shared with the benchmark"],
            "claim": "paper-method reconstruction on a different acquisition, not original-experiment replication",
            "no_outcome_driven_model_changes": True,
        }
    if recipe in ("budget48-direct", "budget48-direct-1500"):
        identity["recipe_id"] = BUDGET48_RECIPE
        identity["operator"].update(grid_size=48, quadrature="midpoint", nodes_per_cell=1,
                                    integration_points=48**3)
        identity["reference"]["scene_budget_adaptation"] = "user_selected_G48_midpoint_20260920"
    if recipe == "budget48-direct":
        identity["recipe_id"] = BUDGET150_RECIPE
        identity["optimizer"]["cosine_max_epochs"] = COMPARISON_EPOCHS
        identity["monitoring"].pop("paper_epochs")
        identity["monitoring"].update(
            plateau="disabled_fixed_comparison_epoch_budget", comparison_epochs=COMPARISON_EPOCHS)
        identity["reference"]["epoch_budget_adaptation"] = "user_selected_150_full_training_passes_20260921"
        identity["fidelity"]["specified"].remove("1500 epochs")
        identity["fidelity"]["benchmark_settings"].append(
            "150 full training passes with 150-epoch cosine schedule; paper specifies 1500")
    return identity


def _operational_settings(args: argparse.Namespace) -> dict[str, int | str]:
    return {
        "device": str(args.device),
        "neural_point_tile": int(args.neural_point_tile),
        "renderer_point_tile": int(args.renderer_point_tile),
        "pair_tile": int(args.pair_tile),
    }


def _validate_cli_recipe(args: argparse.Namespace) -> None:
    recipe = getattr(args, "recipe", "legacy-midpoint")
    max_epochs = _recipe_identity(recipe)["optimizer"]["cosine_max_epochs"]
    if args.epochs < 1 or args.epochs > max_epochs:
        raise ValueError(f"--epochs must be in [1,{max_epochs}]")
    expected_grid = 48 if recipe in ("legacy-midpoint", "budget48-direct", "budget48-direct-1500") else PARENT_GRID
    if args.grid_size != expected_grid:
        raise ValueError(
            f"training quadrature for {recipe} is frozen at G={expected_grid}; "
            "use its matching recipe when resuming")
    if not (1 <= args.neural_point_tile <= 4096):
        raise ValueError("--neural-point-tile must be in [1,4096]")
    if not (1 <= args.renderer_point_tile <= 65536):
        raise ValueError("--renderer-point-tile must be in [1,65536]")
    if not (1 <= args.pair_tile <= 16):
        raise ValueError("--pair-tile must be in [1,16]")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    if args.device == "cpu" and not args.allow_cpu_validation:
        raise ValueError(
            "CPU is reserved for focused validation; pass --allow-cpu-validation explicitly. "
            "Do not use this entry point for login-node training.")
    _finite_positive(args.host_rss_limit_gib, "--host-rss-limit-gib")


def _validate_canonical_contract(contract: Mapping[str, Any]) -> None:
    from rift.rift_dataset import collection_contract
    if collection_contract(contract) is not None:
        _recipe_identity(num_train=len(contract["role_ids"]["train"]))
        return
    roles = contract.get("role_ids")
    if not isinstance(roles, Mapping):
        raise ValueError("sealed B787 contract lacks role_ids")
    counts = (
        len(roles.get("train", ())),
        len(roles.get("validation", ())),
        len(roles.get("reserved_test", ())),
        len(roles.get("unused", ())),
    )
    expected = (
        CANONICAL_TRAIN_COUNT,
        CANONICAL_VALIDATION_COUNT,
        CANONICAL_TEST_COUNT,
        CANONICAL_UNUSED_COUNT,
    )
    if counts != expected:
        raise ValueError(f"SpINR-style B787 requires canonical role counts {expected}, got {counts}")
    if tuple(contract.get("response_shape", ())) != (10000, 16, 16, 1, 600):
        raise ValueError("SpINR-style B787 requires response shape [10000,16,16,1,600]")
    if contract.get("response_dtype") != "complex64":
        raise ValueError("SpINR-style B787 requires complex64 raw responses")
    access = contract.get("response_access")
    if not isinstance(access, Mapping) or access.get("reserved_test_materialized") or access.get("unused_materialized"):
        raise ValueError("SpINR-style B787 requires sealed test and unused response access")


def _finite_nonnegative_or_infinity(value: object, label: str) -> float:
    value = float(value)
    if math.isnan(value) or value < 0:
        raise ValueError(f"{label} must be nonnegative or +inf, got {value!r}")
    return value


def _as_exact_nonnegative_step(value: object, label: str) -> int:
    """Read an Adam step counter without accepting a fractional/tampered value."""

    if torch.is_tensor(value):
        if value.numel() != 1 or not torch.isfinite(value).all():
            raise ValueError(f"{label} must be one finite scalar")
        value = float(value.detach().cpu().item())
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a nonnegative integer")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0 or int(numeric) != numeric:
        raise ValueError(f"{label} must be a nonnegative integer")
    return int(numeric)


def _spinr_model_state_layout() -> dict[str, tuple[int, ...]]:
    """Return fixed model-state names in the optimizer's parameter order."""

    expected_shapes: dict[str, tuple[int, ...]] = {}
    for layer_index in range(SPINR_STYLE_HIDDEN_LAYERS):
        in_features = SPINR_STYLE_INPUT_FEATURES if layer_index == 0 else SPINR_STYLE_HIDDEN_WIDTH
        expected_shapes[f"hidden.{layer_index}.weight"] = (SPINR_STYLE_HIDDEN_WIDTH, in_features)
        expected_shapes[f"hidden.{layer_index}.bias"] = (SPINR_STYLE_HIDDEN_WIDTH,)
    expected_shapes["head.weight"] = (1, SPINR_STYLE_HIDDEN_WIDTH)
    expected_shapes["head.bias"] = (1,)
    return expected_shapes


def _validate_spinr_model_state_dict(state: object) -> None:
    """Validate the fixed FP32 MLP layout before any raw B787 data is opened.

    Instantiating the model here would consume the process RNG before a resume
    restores it.  The released architecture has no buffers, so its named
    parameter layout can be checked directly and without mutating any RNG.
    """

    if not isinstance(state, Mapping):
        raise ValueError("SpINR-style checkpoint lacks a model-state mapping")
    expected_shapes = _spinr_model_state_layout()
    if set(state) != set(expected_shapes):
        raise ValueError("SpINR-style checkpoint model keys disagree with the fixed MLP")
    parameter_count = 0
    for name, expected_shape in expected_shapes.items():
        value = state[name]
        if (not torch.is_tensor(value) or value.dtype != torch.float32
                or tuple(value.shape) != expected_shape):
            raise ValueError(f"SpINR-style checkpoint model tensor {name} has an incompatible layout")
        if not torch.isfinite(value).all():
            raise ValueError(f"SpINR-style checkpoint model tensor {name} must be finite")
        parameter_count += value.numel()
    if parameter_count != SPINR_STYLE_PARAMETER_COUNT:
        raise AssertionError("fixed SpINR-style model-state layout has an unexpected parameter count")


def _checkpoint_tree_equal(left: object, right: object) -> bool:
    """Exact equality for normal checkpoint primitives and tensors."""

    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return isinstance(left, np.ndarray) and isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping) and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_checkpoint_tree_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            isinstance(left, (list, tuple)) and isinstance(right, (list, tuple))
            and len(left) == len(right)
            and all(_checkpoint_tree_equal(a, b) for a, b in zip(left, right))
        )
    return type(left) is type(right) and bool(left == right)


def _optimizer_state_matches_pending_finalization(
    latest_state: Mapping[str, Any],
    finalized_state: Mapping[str, Any],
) -> bool:
    """Compare Adam state across the one scheduler-only finalization transition."""

    latest_optimizer = latest_state.get("optimizer_state_dict")
    finalized_optimizer = finalized_state.get("optimizer_state_dict")
    if not isinstance(latest_optimizer, Mapping) or not isinstance(finalized_optimizer, Mapping):
        return False
    if not _checkpoint_tree_equal(latest_optimizer.get("state"), finalized_optimizer.get("state")):
        return False
    latest_groups = latest_optimizer.get("param_groups")
    finalized_groups = finalized_optimizer.get("param_groups")
    if not (isinstance(latest_groups, list) and isinstance(finalized_groups, list)
            and len(latest_groups) == len(finalized_groups) == 1
            and isinstance(latest_groups[0], Mapping) and isinstance(finalized_groups[0], Mapping)):
        return False
    latest_without_lr = {key: value for key, value in latest_groups[0].items() if key != "lr"}
    finalized_without_lr = {key: value for key, value in finalized_groups[0].items() if key != "lr"}
    return _checkpoint_tree_equal(latest_without_lr, finalized_without_lr)


def _is_expected_pending_finalization_successor(
    latest_state: Mapping[str, Any],
    candidate_state: Mapping[str, Any],
) -> bool:
    """Recognize the sole recoverable artifact window after a pending latest.

    The selected/milestone file is written after scheduler/validation but
    before ``checkpoint_latest``.  It is safe to retain only when it is the
    exact next committed epoch derived from that pending latest—never merely
    because its epoch number happens to be newer.
    """

    latest_execution = latest_state.get("execution")
    candidate_execution = candidate_state.get("execution")
    latest_history = latest_state.get("history")
    candidate_history = candidate_state.get("history")
    if not (isinstance(latest_execution, Mapping) and isinstance(candidate_execution, Mapping)
            and isinstance(latest_history, list) and isinstance(candidate_history, list)):
        return False
    if latest_execution.get("phase") != "pending_epoch_finalization":
        return False
    latest_epoch = latest_state.get("epoch_index")
    candidate_epoch = candidate_state.get("epoch_index")
    if (isinstance(latest_epoch, bool) or isinstance(candidate_epoch, bool)
            or not isinstance(latest_epoch, int) or not isinstance(candidate_epoch, int)
            or candidate_epoch != latest_epoch + 1):
        return False
    candidate_partial = candidate_execution.get("partial_epoch")
    if (candidate_execution.get("phase") != "updates" or not isinstance(candidate_partial, Mapping)
            or candidate_partial.get("completed_updates") != 0
            or candidate_execution.get("stop_reason") is not None):
        return False
    if (len(candidate_history) != len(latest_history) + 1
            or candidate_history[:-1] != latest_history
            or not isinstance(candidate_history[-1], Mapping)
            or candidate_history[-1].get("epoch") != candidate_epoch):
        return False
    # The training-side values in the committed record are fully determined by
    # the durable partial epoch.  Check them before accepting the narrow
    # atomic-write window, rather than allowing a syntactically valid future
    # artifact to invent a different streamed training result.  Validation
    # values themselves are checked again against a replayed finalization
    # after authorized views have been materialized.
    latest_partial = latest_execution.get("partial_epoch")
    latest_optimizer = latest_state.get("optimizer_state_dict")
    latest_groups = latest_optimizer.get("param_groups") if isinstance(latest_optimizer, Mapping) else None
    record = candidate_history[-1]
    if not (
        isinstance(latest_partial, Mapping)
        and isinstance(latest_groups, list)
        and len(latest_groups) == 1
        and isinstance(latest_groups[0], Mapping)
    ):
        return False
    completed_updates = latest_partial.get("completed_updates")
    if isinstance(completed_updates, bool) or not isinstance(completed_updates, int) or completed_updates <= 0:
        return False
    expected_record_values = {
        "streamed_train_native_objective": float(latest_partial["loss_sum"]) / completed_updates,
        "mean_unclipped_gradient_norm": float(latest_partial["gradient_norm_sum"]) / completed_updates,
        "fraction_clipped_updates": float(latest_partial["clipped_update_count"]) / completed_updates,
        "epoch_seconds": float(latest_partial["elapsed_seconds"]),
        "learning_rate": float(latest_groups[0]["lr"]),
    }
    for key, expected_value in expected_record_values.items():
        try:
            actual_value = float(record[key])
        except (KeyError, TypeError, ValueError):
            return False
        if not math.isfinite(actual_value) or actual_value != expected_value:
            return False
    for key in ("model_state_dict", "rng_state", "normalization", "sealed_npz_protocol_contract",
                "acquisition_identity", "spinr_style_recipe", "operational_settings"):
        if not _checkpoint_tree_equal(latest_state.get(key), candidate_state.get(key)):
            return False
    return _optimizer_state_matches_pending_finalization(latest_state, candidate_state)


def _committed_artifact_matches_history_prefix(
    artifact: Mapping[str, Any],
    latest_state: Mapping[str, Any],
    *,
    epoch: int,
) -> bool:
    """Check a normal selected/milestone artifact against latest history."""

    history = latest_state.get("history")
    execution = artifact.get("execution")
    partial = execution.get("partial_epoch") if isinstance(execution, Mapping) else None
    return (
        isinstance(history, list)
        and artifact.get("epoch_index") == epoch
        and isinstance(execution, Mapping)
        and execution.get("phase") == "updates"
        and execution.get("stop_reason") is None
        and isinstance(partial, Mapping)
        and partial.get("completed_updates") == 0
        and artifact.get("history") == history[:epoch]
        and _checkpoint_tree_equal(artifact.get("normalization"), latest_state.get("normalization"))
    )


def _replayed_pending_successor_matches(
    artifact: Mapping[str, Any],
    expected_state: Mapping[str, Any],
) -> bool:
    """Bind an ahead selected/milestone artifact to the actual replay result.

    A selected/milestone file can be durable after a fully completed epoch but
    before ``checkpoint_latest`` has been replaced.  It is not authoritative:
    the durable pending latest remains the recovery source.  Once the runner
    deterministically replays scheduler, diagnostics, and validation, every
    scientific state field of that ahead artifact must agree.  Runtime tile
    settings are deliberately excluded here: they were already required to
    match the pending latest and may be changed only for the resumed process.
    """

    excluded = {"operational_settings"}
    artifact_keys = set(artifact).difference(excluded)
    expected_keys = set(expected_state).difference(excluded)
    if artifact_keys != expected_keys:
        return False
    return all(
        _checkpoint_tree_equal(artifact[key], expected_state[key])
        for key in expected_keys
    )


def _expected_cosine_learning_rate(epoch_index: int, max_epochs: int | None = None) -> float:
    """Analytic CosineAnnealingLR value after exactly ``epoch_index`` steps."""

    return 1.0e-5 + (1.0e-4 - 1.0e-5) * (
        1.0 + math.cos(math.pi * float(epoch_index) / float(
            CANONICAL_MAX_EPOCHS if max_epochs is None else max_epochs))) / 2.0


def _validate_complete_rng_state_without_changing_runtime(state: object) -> None:
    """Exercise strict RNG restoration while returning every generator unchanged."""

    before = capture_rng_state()
    try:
        restore_rng_state(state, require_complete=True)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("SpINR-style resume requires a complete usable RNG state") from exc
    finally:
        # ``before`` was just captured from this same runtime, so a failure to
        # restore it is a local runtime fault rather than a checkpoint policy
        # that can be ignored.
        restore_rng_state(before, require_complete=True)


def _validate_resume_checkpoint_structure(
    checkpoint: Mapping[str, Any],
    *,
    recipe_identity: Mapping[str, Any],
) -> None:
    """Reject an invalid continuation before data/model mutation or payload access."""

    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("--resume is not a SpINR-style B787 checkpoint")
    saved_recipe = checkpoint.get("spinr_style_recipe")
    if saved_recipe != recipe_identity and saved_recipe != _legacy_batch_metadata(recipe_identity):
        raise ValueError("SpINR-style resume would change a scientific recipe setting")
    schedule_epochs = recipe_identity["optimizer"]["cosine_max_epochs"]
    updates_per_epoch = recipe_identity["batching"]["updates_per_epoch"]
    if not isinstance(checkpoint.get("sealed_npz_protocol_contract"), Mapping):
        raise ValueError("SpINR-style resume requires a complete sealed NPZ protocol contract")
    if not isinstance(checkpoint.get("acquisition_identity"), Mapping):
        raise ValueError("SpINR-style resume requires a complete acquisition identity")
    normalization = checkpoint.get("normalization")
    selection = checkpoint.get("selection")
    execution = checkpoint.get("execution")
    history = checkpoint.get("history")
    if not isinstance(normalization, Mapping) or not isinstance(selection, Mapping):
        raise ValueError("SpINR-style checkpoint lacks normalization or selection state")
    if not isinstance(execution, Mapping) or not isinstance(history, list):
        raise ValueError("SpINR-style checkpoint lacks execution or history state")
    _finite_positive(normalization.get("training_mean_raw_power"), "saved training mean raw power")
    _finite_positive(normalization.get("initial_output_scale"), "saved initial output scale")
    _finite_positive(normalization.get("initial_scale_observed_energy"), "saved observed scale energy")
    _finite_positive(normalization.get("initial_scale_predicted_energy"), "saved predicted scale energy")
    initial_ids = normalization.get("initial_scale_training_ids")
    if not isinstance(initial_ids, list) or len(initial_ids) != CANONICAL_INIT_SCALE_COUNT:
        raise ValueError("SpINR-style resume has an invalid fixed initial-scale role list")
    if any(isinstance(item, bool) or int(item) != item for item in initial_ids):
        raise ValueError("SpINR-style initial-scale role IDs must be integers")
    epoch_index = checkpoint.get("epoch_index")
    next_update_index = checkpoint.get("next_update_index")
    if (isinstance(epoch_index, bool) or isinstance(next_update_index, bool)
            or not isinstance(epoch_index, int) or not isinstance(next_update_index, int)
            or epoch_index < 0 or next_update_index < 0
            or next_update_index > updates_per_epoch):
        raise ValueError("SpINR-style checkpoint has an invalid epoch/update cursor")
    if execution.get("schema") != EXECUTION_SCHEMA:
        raise ValueError("SpINR-style checkpoint has an incompatible execution schema")
    phase = execution.get("phase")
    if phase not in {"updates", "pending_epoch_finalization", "completed"}:
        raise ValueError("SpINR-style checkpoint has an invalid execution phase")
    partial_epoch = execution.get("partial_epoch")
    if not isinstance(partial_epoch, Mapping):
        raise ValueError("SpINR-style checkpoint lacks partial-epoch accumulators")
    completed_updates = partial_epoch.get("completed_updates")
    if (isinstance(completed_updates, bool) or not isinstance(completed_updates, int)
            or completed_updates != next_update_index):
        raise ValueError("SpINR-style partial-epoch update count disagrees with its cursor")
    if phase == "updates" and not (0 <= completed_updates < updates_per_epoch):
        raise ValueError("SpINR-style updates phase has an invalid update cursor")
    if phase == "pending_epoch_finalization" and completed_updates != updates_per_epoch:
        raise ValueError("SpINR-style pending finalization must contain one complete training epoch")
    if phase == "completed" and completed_updates != 0:
        raise ValueError("SpINR-style completed state must not retain a partial epoch")
    if recipe_identity.get("recipe_id") in (PAPER_RECIPE, DIRECT_RECIPE, BUDGET48_RECIPE, BUDGET150_RECIPE):
        expected_coverage = optimization_coverage(
            checkpoint["sealed_npz_protocol_contract"].get("role_ids", {}).get("train", ()),
            epoch_index, completed_updates)
        if not _checkpoint_tree_equal(checkpoint.get("optimization_coverage"), expected_coverage):
            raise ValueError("SpINR optimization coverage disagrees with the committed cursor/roles")
    for key in ("loss_sum", "gradient_norm_sum", "elapsed_seconds"):
        value = float(partial_epoch.get(key, float("nan")))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"SpINR-style partial epoch has invalid {key}")
    clipped_update_count = partial_epoch.get("clipped_update_count")
    if (isinstance(clipped_update_count, bool) or not isinstance(clipped_update_count, int)
            or not 0 <= clipped_update_count <= completed_updates):
        raise ValueError("SpINR-style partial epoch has an invalid clipped-update count")
    if phase in {"updates", "completed"} and completed_updates == 0:
        if (float(partial_epoch["loss_sum"]) != 0.0
                or float(partial_epoch["gradient_norm_sum"]) != 0.0
                or float(partial_epoch["elapsed_seconds"]) != 0.0
                or clipped_update_count != 0):
            raise ValueError("SpINR-style zero-cursor state must not retain partial-epoch accumulators")
    stop_reason = execution.get("stop_reason")
    if phase == "completed":
        if stop_reason not in _TERMINAL_STOP_REASONS:
            raise ValueError("SpINR-style completed state lacks a registered stop reason")
        if recipe_identity.get("recipe_id") in (DIRECT_RECIPE, BUDGET48_RECIPE, BUDGET150_RECIPE):
            expected_stop = "epoch_budget_reached" if epoch_index == schedule_epochs else "development_epoch_budget_reached"
            if stop_reason != expected_stop or epoch_index > schedule_epochs:
                raise ValueError("direct-bin recipe requires its fixed paper budget or an explicit development stop")
    elif stop_reason is not None:
        raise ValueError("SpINR-style resumable state must not carry a terminal stop reason")
    if selection.get("metric") != "full_validation_coherent_relative_mse" or selection.get("tie_policy") != "earliest":
        raise ValueError("SpINR-style checkpoint has incompatible selection semantics")
    best_metric = _finite_nonnegative_or_infinity(
        selection.get("best_validation_rel_mse"), "saved best validation RelMSE")
    best_epoch = selection.get("best_epoch")
    if best_epoch is not None and (isinstance(best_epoch, bool) or not isinstance(best_epoch, int) or best_epoch < 1):
        raise ValueError("SpINR-style checkpoint has an invalid best epoch")
    if not all(isinstance(record, Mapping) for record in history):
        raise ValueError("SpINR-style checkpoint history must contain records")
    if len(history) != epoch_index:
        raise ValueError("SpINR-style history must contain exactly the committed epoch records")
    validation_candidates: list[tuple[int, float]] = []
    running_train_diagnostic = math.inf
    running_validation = math.inf
    for expected_epoch, record in enumerate(history, start=1):
        record_epoch = record.get("epoch")
        if (isinstance(record_epoch, bool) or not isinstance(record_epoch, int)
                or record_epoch != expected_epoch):
            raise ValueError("SpINR-style history epochs must be contiguous and one-based")
        if recipe_identity.get("recipe_id") in (PAPER_RECIPE, DIRECT_RECIPE, BUDGET48_RECIPE, BUDGET150_RECIPE):
            expected = optimization_coverage(
                checkpoint["sealed_npz_protocol_contract"]["role_ids"]["train"],
                expected_epoch, 0, include_per_view=False)
            if not _checkpoint_tree_equal(record.get("optimization_coverage"), expected):
                raise ValueError("SpINR history optimization coverage is inconsistent")
        for key in ("streamed_train_native_objective", "mean_unclipped_gradient_norm", "epoch_seconds"):
            value = _finite_nonnegative_or_infinity(record.get(key), f"saved {key}")
            if not math.isfinite(value):
                raise ValueError(f"SpINR-style history {key} must be finite")
        clip_fraction = float(record.get("fraction_clipped_updates", float("nan")))
        if not math.isfinite(clip_fraction) or not 0.0 <= clip_fraction <= 1.0:
            raise ValueError("SpINR-style history has an invalid clipped-update fraction")
        record_lr = _finite_positive(record.get("learning_rate"), "saved epoch learning rate")
        expected_record_lr = _expected_cosine_learning_rate(expected_epoch - 1, schedule_epochs)
        if not math.isclose(record_lr, expected_record_lr, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("SpINR-style history learning rate disagrees with its cosine schedule")
        train_diagnostic = record.get("train_diagnostic")
        validation = record.get("validation")
        if (train_diagnostic is None) != (validation is None):
            raise ValueError("SpINR-style diagnostics and validation must be committed together")
        if validation is None:
            if expected_epoch % CANONICAL_VALIDATION_EVERY == 0:
                raise ValueError("SpINR-style periodic validation is missing from committed history")
            continue
        if not isinstance(validation, Mapping) or not isinstance(train_diagnostic, Mapping):
            raise ValueError("SpINR-style validation record must contain metric mappings")
        for label, metrics, expected_views in (
            ("train diagnostic", train_diagnostic, recipe_identity["monitoring"]["train_diagnostic_views"]),
            ("validation", validation, CANONICAL_VALIDATION_COUNT),
        ):
            if metrics.get("views") != expected_views:
                raise ValueError(f"SpINR-style {label} view count disagrees with the recipe")
            for metric_key, metric_label in (
                ("coherent_relative_mse", "coherent RelMSE"),
                ("coherent_relative_l2", "coherent RelL2"),
                ("native_spectral_objective", "native objective"),
            ):
                metric_value = _finite_nonnegative_or_infinity(
                    metrics.get(metric_key), f"saved {label} {metric_label}")
                if not math.isfinite(metric_value):
                    raise ValueError(f"SpINR-style {label} {metric_label} must be finite")
        train_metric = float(train_diagnostic["coherent_relative_mse"])
        validation_metric = float(validation["coherent_relative_mse"])
        running_train_diagnostic = min(running_train_diagnostic, train_metric)
        running_validation = min(running_validation, validation_metric)
        for key, expected_value in (
            ("running_best_train_diagnostic_rel_mse", running_train_diagnostic),
            ("running_best_validation_rel_mse", running_validation),
        ):
            actual_value = _finite_nonnegative_or_infinity(record.get(key), f"saved {key}")
            if actual_value != expected_value:
                raise ValueError("SpINR-style running-best metric disagrees with committed history")
        validation_candidates.append((expected_epoch, validation_metric))
    if validation_candidates:
        expected_best_epoch, expected_best_metric = min(validation_candidates, key=lambda item: item[1])
        if best_epoch != expected_best_epoch or best_metric != expected_best_metric:
            raise ValueError("SpINR-style selection state disagrees with committed validation history")
    elif best_epoch is not None or not math.isinf(best_metric):
        raise ValueError("SpINR-style selection state claims validation before any validation record")
    _validate_spinr_model_state_dict(checkpoint.get("model_state_dict"))
    optimizer_state = checkpoint.get("optimizer_state_dict")
    scheduler_state = checkpoint.get("scheduler_state_dict")
    if not isinstance(optimizer_state, Mapping) or not isinstance(scheduler_state, Mapping):
        raise ValueError("SpINR-style checkpoint lacks optimizer or scheduler state")
    groups = optimizer_state.get("param_groups")
    if not isinstance(groups, list) or len(groups) != 1 or not isinstance(groups[0], Mapping):
        raise ValueError("SpINR-style checkpoint has incompatible Adam parameter groups")
    group = groups[0]
    if (tuple(group.get("betas", ())) != (0.9, 0.999)
            or float(group.get("eps", float("nan"))) != 1e-8
            or float(group.get("weight_decay", float("nan"))) != 0.0
            or group.get("maximize", False) is not False
            or group.get("amsgrad", False) is not False
            or group.get("differentiable", False) is not False
            or group.get("capturable", False) is not False
            or group.get("decoupled_weight_decay", False) is not False
            or group.get("foreach", None) is not None
            or group.get("fused", None) not in (None, False)):
        raise ValueError("SpINR-style resume would change Adam hyperparameters")
    param_ids = group.get("params")
    if (not isinstance(param_ids, list) or not param_ids
            or len(set(param_ids)) != len(param_ids)):
        raise ValueError("SpINR-style checkpoint has invalid Adam parameter membership")
    model_state = checkpoint["model_state_dict"]
    if not isinstance(model_state, Mapping):
        raise AssertionError("validated SpINR-style model state unexpectedly lost its mapping")
    model_tensors = [model_state[name] for name in _spinr_model_state_layout()]
    if len(param_ids) != len(model_tensors):
        raise ValueError("SpINR-style Adam parameter membership disagrees with the model state")
    if param_ids != list(range(len(model_tensors))):
        raise ValueError("SpINR-style Adam parameter order disagrees with the fixed MLP")
    if (int(scheduler_state.get("T_max", -1)) != schedule_epochs
            or float(scheduler_state.get("eta_min", float("nan"))) != 1e-5):
        raise ValueError("SpINR-style resume would change cosine scheduler settings")
    scheduler_epoch = scheduler_state.get("last_epoch")
    if (isinstance(scheduler_epoch, bool) or not isinstance(scheduler_epoch, int)
            or scheduler_epoch != epoch_index):
        raise ValueError("SpINR-style scheduler epoch disagrees with committed epoch state")
    expected_lr = _expected_cosine_learning_rate(epoch_index, schedule_epochs)
    group_lr = _finite_positive(group.get("lr"), "saved Adam learning rate")
    if not math.isclose(group_lr, expected_lr, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("SpINR-style Adam learning rate disagrees with its cosine schedule")
    base_lrs = scheduler_state.get("base_lrs")
    last_lrs = scheduler_state.get("_last_lr")
    if (not isinstance(base_lrs, list) or not isinstance(last_lrs, list)
            or len(base_lrs) != 1 or len(last_lrs) != 1
            or not math.isclose(float(base_lrs[0]), 1e-4, rel_tol=0.0, abs_tol=1e-15)
            or not math.isclose(float(last_lrs[0]), expected_lr, rel_tol=0.0, abs_tol=1e-15)):
        raise ValueError("SpINR-style cosine scheduler LR state is incompatible with the recipe")
    step_count = scheduler_state.get("_step_count")
    if (isinstance(step_count, bool) or not isinstance(step_count, int)
            or step_count != epoch_index + 1):
        raise ValueError("SpINR-style cosine scheduler step count is inconsistent with epoch state")
    total_updates = epoch_index * updates_per_epoch + completed_updates
    adam_states = optimizer_state.get("state")
    if not isinstance(adam_states, Mapping):
        raise ValueError("SpINR-style checkpoint lacks Adam moment state")
    if total_updates == 0:
        if adam_states:
            raise ValueError("SpINR-style fresh optimizer state must not contain moment updates")
    else:
        if set(adam_states) != set(param_ids):
            raise ValueError("SpINR-style Adam moments must cover every trainable parameter")
        for param_id, parameter_tensor in zip(param_ids, model_tensors):
            state = adam_states[param_id]
            if not isinstance(state, Mapping):
                raise ValueError("SpINR-style Adam parameter state must be a mapping")
            if _as_exact_nonnegative_step(state.get("step"), "saved Adam step") != total_updates:
                raise ValueError("SpINR-style Adam step disagrees with completed logical updates")
            for key in ("exp_avg", "exp_avg_sq"):
                moment = state.get(key)
                if (not torch.is_tensor(moment) or moment.shape != parameter_tensor.shape
                        or moment.dtype != parameter_tensor.dtype):
                    raise ValueError(f"SpINR-style Adam {key} has incompatible parameter shape")
                if not torch.is_floating_point(moment) or not torch.isfinite(moment).all():
                    raise ValueError(f"SpINR-style Adam {key} must be finite floating point")
            if "max_exp_avg_sq" in state:
                raise ValueError("SpINR-style Adam state unexpectedly contains AMSGrad moments")
    _validate_complete_rng_state_without_changing_runtime(checkpoint.get("rng_state"))


def _validate_normalization_against_sealed_contract(
    normalization: object,
    sealed_contract: Mapping[str, Any],
) -> None:
    """Ensure an artifact's fixed-scale rows are the authorized train prefix."""

    if not isinstance(normalization, Mapping):
        raise ValueError("SpINR-style checkpoint lacks normalization state")
    raw_ids = normalization.get("initial_scale_training_ids")
    role_ids = sealed_contract.get("role_ids")
    train_ids = role_ids.get("train") if isinstance(role_ids, Mapping) else None
    if not isinstance(raw_ids, list) or not isinstance(train_ids, (list, tuple)):
        raise ValueError("SpINR-style checkpoint lacks authorized initial-scale row IDs")
    try:
        saved_ids = tuple(int(item) for item in raw_ids)
        expected_ids = tuple(int(item) for item in train_ids[:CANONICAL_INIT_SCALE_COUNT])
    except (TypeError, ValueError) as exc:
        raise ValueError("SpINR-style initial-scale row IDs are malformed") from exc
    if saved_ids != expected_ids:
        raise ValueError("SpINR-style checkpoint would change the fixed initial-scale training rows")


def _load_resume_preflight(
    resume: str | None,
    *,
    recipe_identity: Mapping[str, Any],
    requested_epochs: int,
) -> Mapping[str, Any] | None:
    if resume is None:
        return None
    checkpoint = load_tensor_checkpoint(resume, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("--resume did not contain a checkpoint mapping")
    _validate_resume_checkpoint_structure(checkpoint, recipe_identity=recipe_identity)
    execution = checkpoint["execution"]
    if not isinstance(execution, Mapping):
        raise AssertionError("validated resume checkpoint unexpectedly lacks execution state")
    if execution.get("phase") == "completed":
        raise ValueError("SpINR-style completed runs are terminal and cannot be resumed")
    completed_epoch = int(checkpoint["epoch_index"])
    if completed_epoch > int(requested_epochs):
        raise ValueError("SpINR-style resume already exceeds the requested epoch budget")
    partial_epoch = execution.get("partial_epoch")
    has_partial_next_epoch = (
        execution.get("phase") == "pending_epoch_finalization"
        or (execution.get("phase") == "updates"
            and isinstance(partial_epoch, Mapping)
            and int(partial_epoch["completed_updates"]) > 0)
    )
    if has_partial_next_epoch and completed_epoch + 1 > int(requested_epochs):
        raise ValueError(
            "SpINR-style partial next epoch would exceed the requested epoch budget; "
            "resume with the original or a larger budget")
    return checkpoint


def _resolve_output_namespace(args: argparse.Namespace, resume_checkpoint: Mapping[str, Any] | None) -> Path:
    """Protect a fresh identity and reject unsupported output relocation."""

    root = Path(args.checkpoint_root).expanduser().resolve()
    requested = Path(args.checkpoint_name)
    if requested.is_absolute() or len(requested.parts) != 1 or requested.name in {"", ".", ".."}:
        raise ValueError("--checkpoint-name must be one new artifact identity below --checkpoint-root")
    checkpoint_dir = (root / requested).resolve()
    if root != checkpoint_dir and root not in checkpoint_dir.parents:
        raise ValueError("--checkpoint-name escapes --checkpoint-root")
    if resume_checkpoint is None:
        if checkpoint_dir.exists() and not checkpoint_dir.is_dir():
            raise FileExistsError(f"fresh SpINR-style checkpoint target is not a directory: {checkpoint_dir}")
        if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
            raise FileExistsError(
                f"fresh SpINR-style run refuses nonempty checkpoint namespace {checkpoint_dir}; "
                "choose a new artifact identity instead of overwriting existing results")
        return checkpoint_dir
    resume_path = Path(args.resume).expanduser().resolve()
    if resume_path.name != "checkpoint_latest.pth.tar" or resume_path.parent != checkpoint_dir:
        raise ValueError(
            "SpINR-style continuation must resume checkpoint_latest.pth.tar in its original "
            "checkpoint namespace; output relocation is not supported")
    if not checkpoint_dir.is_dir():
        raise ValueError("SpINR-style resume checkpoint namespace is not a directory")
    unexpected = sorted(
        path.name for path in checkpoint_dir.iterdir()
        if path.name not in _CHECKPOINT_FILENAMES and not _is_managed_checkpoint_temp(path)
    )
    if unexpected:
        raise ValueError(
            "SpINR-style checkpoint namespace contains unexpected files; refusing to mix artifacts: "
            + ", ".join(unexpected))
    if (checkpoint_dir / "checkpoint_final.pth.tar").is_file():
        raise ValueError("SpINR-style checkpoint namespace already has a terminal final artifact")
    return checkpoint_dir


def preflight_b787_development_inputs(
    *,
    npz_path: str,
    manifest_path: str,
    resume_checkpoint: Mapping[str, Any] | None,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Bind headers, roles, and physical acquisition inputs without response access.

    The caller must separately invoke :func:`materialize_b787_development_views`
    after all recipe, namespace, and acquisition checks pass.  Keeping this
    split explicit makes payload-free continuation rejection mechanically
    testable.
    """

    from rift.rift_dataset import collection_manifest, load_object_contract
    if collection_manifest(manifest_path):
        arrays, contract = load_object_contract(npz_path, manifest_path)
    else:
        arrays, contract = _load_sealed_npz_protocol_contract(
            npz_path, manifest_path, num_train=CANONICAL_TRAIN_COUNT,
            num_val=CANONICAL_VALIDATION_COUNT, num_test=CANONICAL_TEST_COUNT)
    _validate_canonical_contract(contract)
    acquisition_identity = build_spinr_style_acquisition_identity(arrays, contract)
    if resume_checkpoint is not None:
        _validate_saved_sealed_npz_protocol_contract(
            resume_checkpoint.get("sealed_npz_protocol_contract"), contract)
        validate_spinr_style_acquisition_identity(
            resume_checkpoint.get("acquisition_identity"), acquisition_identity)
        _validate_normalization_against_sealed_contract(
            resume_checkpoint.get("normalization"), contract)
    return arrays, contract, acquisition_identity


def materialize_b787_development_views(
    arrays: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> SealedRawComplexViews:
    """Perform the first permitted raw-response operation after all preflight."""

    # This adapter is capability-limited to the authorized train/validation
    # IDs.  The response payload is intentionally unreachable before here.
    return build_sealed_raw_complex_views(arrays, contract)


def load_b787_development_views(
    *,
    npz_path: str,
    manifest_path: str,
    resume_checkpoint: Mapping[str, Any] | None,
) -> tuple[SealedRawComplexViews, Mapping[str, Any]]:
    """Compatibility wrapper for focused callers that need materialized views.

    Production :func:`run` uses the split preflight/materialization functions
    above so it can validate its output namespace before raw bytes are opened.
    """

    arrays, contract, _acquisition_identity = preflight_b787_development_inputs(
        npz_path=npz_path,
        manifest_path=manifest_path,
        resume_checkpoint=resume_checkpoint,
    )
    return materialize_b787_development_views(arrays, contract), contract


def _validate_checkpoint_namespace_artifacts(
    *,
    checkpoint_dir: Path,
    resume_checkpoint: Mapping[str, Any] | None,
    sealed_contract: Mapping[str, Any],
    acquisition_identity: Mapping[str, Any],
    recipe_identity: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    """Validate artifacts before raw data, returning only recoverable ahead files.

    The return value is intentionally limited to a selected/milestone artifact
    written in the one permitted pending-finalization window.  The caller must
    later bind each such artifact to the replayed finalization before it can
    remain in the namespace.
    """

    if resume_checkpoint is None:
        return {}
    latest_path = checkpoint_dir / "checkpoint_latest.pth.tar"
    if not latest_path.is_file():
        raise ValueError("SpINR-style resume namespace is missing checkpoint_latest.pth.tar")
    artifacts: dict[str, Mapping[str, Any]] = {"checkpoint_latest.pth.tar": resume_checkpoint}
    for name in ("checkpoint_best.pth.tar", "checkpoint_epoch_150.pth.tar", "checkpoint_final.pth.tar"):
        path = checkpoint_dir / name
        if path.exists():
            artifact = load_tensor_checkpoint(path, map_location="cpu")
            if not isinstance(artifact, Mapping):
                raise ValueError(f"SpINR-style artifact {name} is not a checkpoint mapping")
            artifacts[name] = artifact
    for name, artifact in artifacts.items():
        _validate_resume_checkpoint_structure(artifact, recipe_identity=recipe_identity)
        _validate_saved_sealed_npz_protocol_contract(
            artifact.get("sealed_npz_protocol_contract"), sealed_contract)
        validate_spinr_style_acquisition_identity(
            artifact.get("acquisition_identity"), acquisition_identity)
        _validate_normalization_against_sealed_contract(artifact.get("normalization"), sealed_contract)
        if name == "checkpoint_final.pth.tar":
            raise ValueError("SpINR-style final artifact is terminal and cannot be resumed")

    recoverable_successors: dict[str, Mapping[str, Any]] = {}

    selection = resume_checkpoint["selection"]
    if not isinstance(selection, Mapping):
        raise AssertionError("validated latest checkpoint unexpectedly lacks selection state")
    latest_epoch = int(resume_checkpoint["epoch_index"])
    best_epoch = selection["best_epoch"]
    best = artifacts.get("checkpoint_best.pth.tar")
    if best_epoch is None:
        # A future selected artifact can exist only in the narrow atomic-write
        # window after finalizing the pending next epoch but before replacing
        # latest.  A best file beside a normal no-validation latest is stale
        # or mixed evidence, not a resumable state.
        if best is not None:
            future_selection = best.get("selection")
            if not (
                _is_expected_pending_finalization_successor(resume_checkpoint, best)
                and isinstance(future_selection, Mapping)
                and future_selection.get("best_epoch") == latest_epoch + 1
            ):
                raise ValueError(
                    "SpINR-style checkpoint_best is not the exact recoverable successor "
                    "of a no-selection latest checkpoint")
            recoverable_successors["checkpoint_best.pth.tar"] = best
    else:
        if best is None:
            raise ValueError(
                "SpINR-style latest selection requires its committed checkpoint_best artifact; "
                "refusing to continue after a lost selected model")
        best_artifact_epoch = int(best.get("epoch_index", -1))
        if best_artifact_epoch == int(best_epoch):
            if not (
                _committed_artifact_matches_history_prefix(
                    best, resume_checkpoint, epoch=int(best_epoch))
                and _checkpoint_tree_equal(best.get("selection"), selection)
            ):
                raise ValueError(
                    "SpINR-style committed checkpoint_best disagrees with latest selection history")
        else:
            future_selection = best.get("selection")
            if not (
                _is_expected_pending_finalization_successor(resume_checkpoint, best)
                and isinstance(future_selection, Mapping)
                and future_selection.get("best_epoch") == latest_epoch + 1
            ):
                raise ValueError(
                    "SpINR-style checkpoint_best is not the exact recoverable successor "
                    "of its latest checkpoint")
            recoverable_successors["checkpoint_best.pth.tar"] = best

    history = resume_checkpoint["history"]
    if not isinstance(history, list):
        raise AssertionError("validated latest checkpoint unexpectedly lacks history")
    milestone = artifacts.get("checkpoint_epoch_150.pth.tar")
    has_committed_milestone = any(record.get("epoch") == CANONICAL_MIN_EPOCHS for record in history)
    if has_committed_milestone:
        if milestone is None or not _committed_artifact_matches_history_prefix(
                milestone, resume_checkpoint, epoch=CANONICAL_MIN_EPOCHS):
            raise ValueError(
                "SpINR-style latest history requires its committed checkpoint_epoch_150 artifact; "
                "refusing to continue after a lost or mixed milestone model")
    elif milestone is not None:
        if not (
            latest_epoch + 1 == CANONICAL_MIN_EPOCHS
            and _is_expected_pending_finalization_successor(resume_checkpoint, milestone)
        ):
            raise ValueError(
                "SpINR-style checkpoint_epoch_150 is not the exact recoverable successor "
                "of its latest checkpoint")
        recoverable_successors["checkpoint_epoch_150.pth.tar"] = milestone

    history_path = checkpoint_dir / "metrics_history.json"
    saved_metrics: object = None
    if not history_path.is_file():
        saved_metrics = None
    else:
        try:
            with history_path.open("r", encoding="utf-8") as handle:
                saved_metrics = json.load(handle)
        except (OSError, json.JSONDecodeError):
            # ``checkpoint_latest`` is the authoritative atomic artifact; an
            # interrupted sidecar write is regenerated only after its sealed
            # data/recipe/acquisition checks have all passed above.
            saved_metrics = None
    if not isinstance(saved_metrics, Mapping) or saved_metrics.get("history") != resume_checkpoint.get("history"):
        _atomic_json_save({"history": [dict(item) for item in resume_checkpoint["history"]]}, history_path)
    # An interrupted atomic write can leave only these exact private temporary
    # names.  They are no longer authoritative once the matching latest state
    # has been validated; foreign files were rejected in _resolve_output_namespace.
    for path in checkpoint_dir.iterdir():
        if _is_managed_checkpoint_temp(path):
            try:
                path.unlink()
            except OSError as exc:
                raise ValueError(f"could not clear stale managed checkpoint temporary {path.name}") from exc
    return recoverable_successors


def epoch_view_batches(train_ids: Sequence[int], *, epoch: int, seed: int = CANONICAL_SEED) -> tuple[tuple[int, ...], ...]:
    """Return the deterministic selected training permutation in batches of four."""

    ids = np.asarray(tuple(int(item) for item in train_ids), dtype=np.int64)
    if (ids.ndim != 1 or not CANONICAL_INIT_SCALE_COUNT <= ids.size <= CANONICAL_TRAIN_COUNT
            or ids.size % CANONICAL_VIEW_BATCH or len(np.unique(ids)) != ids.size):
        raise ValueError("SpINR-style epochs require 32..3200 distinct training IDs in batches of four")
    order = np.random.default_rng(int(seed) + int(epoch)).permutation(ids)
    batches = tuple(tuple(int(item) for item in row) for row in order.reshape(-1, CANONICAL_VIEW_BATCH))
    return batches


def optimization_coverage(train_ids: Sequence[int], completed_epochs: int,
                          completed_updates: int, *, include_per_view: bool = True) -> dict[str, Any]:
    """Recover optimizer exposure counts from the durable deterministic cursor.

    Normalizer/initialization/evaluation reads never count as fitting exposure.
    The completed-epoch cursor and its finalized successor describe
    the same consumed views, so preemption cannot increment exposure twice.
    """
    updates_per_epoch = len(train_ids) // CANONICAL_VIEW_BATCH
    if completed_epochs < 0 or not 0 <= completed_updates <= updates_per_epoch:
        raise ValueError("invalid optimization coverage cursor")
    ids = tuple(int(i) for i in train_ids)
    batches = epoch_view_batches(ids, epoch=completed_epochs)
    visited = {i for batch in batches[:completed_updates] for i in batch}
    counts = [int(completed_epochs) + int(i in visited) for i in ids]
    result = {
        "schema": "spinr_optimization_coverage_v1",
        "source": "committed_updates_and_deterministic_epoch_permutation",
        "normalization_and_evaluation_reads_counted": False,
        "unique_fitted_views": sum(c > 0 for c in counts),
        "view_exposures": sum(counts),
        "optimizer_updates": completed_epochs*updates_per_epoch + completed_updates,
        "min_exposures_per_train_view": min(counts),
        "max_exposures_per_train_view": max(counts),
    }
    if include_per_view:
        result.update(source_ids=list(ids), exposure_counts=counts)
    return result


@torch.no_grad()
def evaluate_neural_field_tiled(
    model: SpinrStyleINR,
    points_m: torch.Tensor,
    *,
    neural_point_tile: int,
) -> torch.Tensor:
    """Evaluate the signed-real field once, without retaining MLP activations."""

    pieces: list[torch.Tensor] = []
    for start in range(0, points_m.shape[0], int(neural_point_tile)):
        field = model(points_m[start:start + int(neural_point_tile)])
        if torch.is_complex(field) or not torch.isfinite(field).all():
            raise FloatingPointError("SpINR-style network emitted a non-finite or complex field")
        pieces.append(field.to(dtype=torch.float64))
    return torch.cat(pieces, dim=0)


def replay_field_cotangent_tiled(
    model: SpinrStyleINR,
    points_m: torch.Tensor,
    field_cotangent: torch.Tensor,
    *,
    neural_point_tile: int,
) -> None:
    """Replay MLP tiles and accumulate the real field cotangent into parameters."""

    if field_cotangent.shape != (points_m.shape[0],):
        raise ValueError("field cotangent must contain one scalar per quadrature point")
    if torch.is_complex(field_cotangent) or not torch.isfinite(field_cotangent).all():
        raise FloatingPointError("field cotangent must be finite and real")
    for start in range(0, points_m.shape[0], int(neural_point_tile)):
        stop = min(points_m.shape[0], start + int(neural_point_tile))
        field = model(points_m[start:stop])
        torch.autograd.backward(field, field_cotangent[start:stop].to(device=field.device, dtype=field.dtype))


@torch.no_grad()
def render_frequency_response(
    *,
    model: SpinrStyleINR,
    points_m: torch.Tensor,
    cell_volume_m3: float | torch.Tensor,
    initial_output_scale: float,
    frequencies_hz: torch.Tensor,
    kvector: torch.Tensor,
    rx_pos_m: torch.Tensor,
    tx_pos_m: torch.Tensor,
    neural_point_tile: int,
    renderer_point_tile: int,
    pair_tile: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render one complete all-frequency, all-pair response from a fixed MLP."""

    field = evaluate_neural_field_tiled(
        model, points_m, neural_point_tile=neural_point_tile)
    weights = scale_field_to_renderer_weights(
        field,
        cell_volume_m3=cell_volume_m3,
        initial_output_scale=initial_output_scale,
    )
    response = range_forward_operator(
        frequencies_hz,
        kvector,
        rx_pos_m,
        tx_pos_m,
        points_m,
        weights,
        phase_sign=SPINR_STYLE_PHASE_SIGN,
        pair_chunk=pair_tile,
        point_chunk=renderer_point_tile,
        compute_dtype=torch.float64,
        range_model=SPINR_STYLE_RANGE_MODEL,
    )
    if response.shape != (len(frequencies_hz), len(rx_pos_m), len(tx_pos_m)) or response.dtype != torch.complex128:
        raise AssertionError("SpINR-style renderer must return the selected [frequency,Rx,Tx] complex128 data")
    if not torch.isfinite(response.real).all() or not torch.isfinite(response.imag).all():
        raise FloatingPointError("SpINR-style renderer emitted non-finite values")
    return response, field


def response_cotangent(
    predicted_frequency: torch.Tensor,
    observed_frequency: torch.Tensor,
    *,
    training_mean_raw_power: float,
    range_bin_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one native spectral loss and its complex response cotangent."""

    detached_prediction = predicted_frequency.detach().requires_grad_(True)
    loss = spinr_style_objective(
        detached_prediction,
        observed_frequency,
        training_mean_raw_power=training_mean_raw_power,
        range_bin_mask=range_bin_mask,
    )
    cotangent, = torch.autograd.grad(loss, detached_prediction)
    if not torch.isfinite(loss) or not torch.isfinite(cotangent.real).all() or not torch.isfinite(cotangent.imag).all():
        raise FloatingPointError("spectral objective or cotangent is non-finite")
    return loss.detach(), cotangent.detach()


def direct_bin_loss_and_field_cotangent(*, field, observed, frequencies_hz, rx_pos_m, tx_pos_m,
                                       points_m, cell_volume_m3, initial_output_scale,
                                       training_mean_raw_power, renderer_point_tile, pair_tile,
                                       support_m=SPINR_STYLE_SUPPORT_M):
    """Differentiate the paper's direct selected-bin graph; no predicted FFT."""
    mask = scene_range_bin_mask(frequencies_hz, rx_pos_m, tx_pos_m, support_m=support_m)
    kwargs = dict(frequencies_hz=frequencies_hz, rx_pos_m=rx_pos_m, tx_pos_m=tx_pos_m,
                  points_m=points_m, cell_volume_m3=cell_volume_m3,
                  initial_output_scale=initial_output_scale, mask=mask,
                  point_tile=renderer_point_tile, pair_tile=pair_tile)
    with torch.no_grad():
        prediction = render_selected_bins(field=field, **kwargs)
    with torch.enable_grad():
        prediction.requires_grad_(True)
        loss = selected_bin_objective(prediction, observed, mask,
                                       training_mean_raw_power=training_mean_raw_power)
        cotangent, = torch.autograd.grad(loss, prediction)
    if not torch.isfinite(loss) or not torch.isfinite(cotangent).all():
        raise FloatingPointError("direct-bin objective or cotangent is non-finite")
    return loss.detach(), selected_bin_field_vjp(bin_cotangent=cotangent, **kwargs)


@torch.no_grad()
def real_field_cotangent_from_response(
    *,
    response_cotangent_frequency: torch.Tensor,
    frequencies_hz: torch.Tensor,
    kvector: torch.Tensor,
    rx_pos_m: torch.Tensor,
    tx_pos_m: torch.Tensor,
    points_m: torch.Tensor,
    cell_volume_m3: float | torch.Tensor,
    initial_output_scale: float,
    renderer_point_tile: int,
    pair_tile: int,
) -> torch.Tensor:
    """Apply A^H then the fixed real-field conversion exactly once.

    ``range_adjoint_operator`` returns the complex adjoint with respect to
    the renderer's integrated weights.  For the real neural field q, the
    required VJP is ``(4*pi)^2 * volume * a_init * Re(A^H g)``.  There is no
    additional factor of two.
    """

    adjoint = range_adjoint_operator(
        frequencies_hz,
        kvector,
        rx_pos_m,
        tx_pos_m,
        points_m,
        response_cotangent_frequency,
        phase_sign=SPINR_STYLE_PHASE_SIGN,
        pair_chunk=pair_tile,
        point_chunk=renderer_point_tile,
        compute_dtype=torch.float64,
        range_model=SPINR_STYLE_RANGE_MODEL,
    )
    volume = torch.as_tensor(cell_volume_m3, device=adjoint.device, dtype=torch.float64)
    if volume.ndim == 0:
        if not torch.isfinite(volume).all() or bool((volume <= 0).any()):
            raise ValueError("cell_volume_m3 must be finite and positive")
    elif volume.shape == adjoint.shape:
        if not torch.isfinite(volume).all() or bool((volume <= 0).any()):
            raise ValueError("cell_volume_m3 entries must be finite and positive")
    else:
        raise ValueError("cell_volume_m3 must be a scalar or one volume per field point")
    scale = ((4.0 * math.pi) ** 2) * float(initial_output_scale)
    cotangent = scale * volume * adjoint.real
    if torch.is_complex(cotangent) or not torch.isfinite(cotangent).all():
        raise FloatingPointError("range adjoint produced a non-finite real field cotangent")
    return cotangent


@torch.no_grad()
def estimate_initial_output_scale(
    *,
    model: SpinrStyleINR,
    views: SealedRawComplexViews,
    points_m: torch.Tensor,
    cell_volume_m3: float,
    frequencies_hz: torch.Tensor,
    kvector: torch.Tensor,
    neural_point_tile: int,
    renderer_point_tile: int,
    pair_tile: int,
    device: torch.device,
) -> tuple[float, tuple[int, ...], float, float]:
    """Set the fixed amplitude conditioning from the first 32 train IDs only."""

    source_ids = tuple(views.role_ids("train")[:CANONICAL_INIT_SCALE_COUNT])
    if len(source_ids) != CANONICAL_INIT_SCALE_COUNT:
        raise AssertionError("canonical B787 contract must expose the first 32 training IDs")
    # q is common across views at initialization.  Use a fixed unity scale
    # here; the returned a_init is the only amplitude conditioning applied.
    field = evaluate_neural_field_tiled(model, points_m, neural_point_tile=neural_point_tile)
    weights = scale_field_to_renderer_weights(
        field, cell_volume_m3=cell_volume_m3, initial_output_scale=1.0)
    predicted_energy = 0.0
    observed_energy = 0.0
    for source_id in source_ids:
        observed, rx_pos, tx_pos = views.tensor_view(source_id, device=device)
        predicted = range_forward_operator(
            frequencies_hz,
            kvector,
            rx_pos,
            tx_pos,
            points_m,
            weights,
            phase_sign=SPINR_STYLE_PHASE_SIGN,
            pair_chunk=pair_tile,
            point_chunk=renderer_point_tile,
            compute_dtype=torch.float64,
            range_model=SPINR_STYLE_RANGE_MODEL,
        )
        predicted_energy += float(predicted.abs().square().sum().item())
        observed_energy += float(observed.abs().square().sum().item())
    _finite_positive(predicted_energy, "random-field predicted energy")
    _finite_positive(observed_energy, "initial-scale observed energy")
    output_scale = 0.1 * math.sqrt(observed_energy / predicted_energy)
    return (
        _finite_positive(output_scale, "initial output scale"),
        source_ids,
        observed_energy,
        predicted_energy,
    )


def _native_metrics(
    *,
    numerator: float,
    denominator: float,
    native_objective_sum: float,
    count: int,
) -> dict[str, float | int]:
    denominator = _finite_positive(denominator, "coherent metric denominator")
    numerator = float(numerator)
    if not math.isfinite(numerator) or numerator < 0:
        raise FloatingPointError("coherent metric numerator must be finite and nonnegative")
    relative_mse = numerator / denominator
    if not math.isfinite(relative_mse):
        raise FloatingPointError("coherent relative MSE must be finite")
    native_objective = float(native_objective_sum) / int(count)
    if not math.isfinite(native_objective) or native_objective < 0:
        raise FloatingPointError("native spectral objective must be finite and nonnegative")
    return {
        "views": int(count),
        "coherent_relative_mse": relative_mse,
        "coherent_relative_l2": math.sqrt(relative_mse),
        "native_spectral_objective": native_objective,
    }


@torch.no_grad()
def evaluate_role(
    *,
    model: SpinrStyleINR,
    views: SealedRawComplexViews,
    role: str,
    source_ids: Sequence[int] | None,
    points_m: torch.Tensor,
    cell_volume_m3: float | torch.Tensor,
    initial_output_scale: float,
    frequencies_hz: torch.Tensor,
    kvector: torch.Tensor,
    training_mean_raw_power: float,
    neural_point_tile: int,
    renderer_point_tile: int,
    pair_tile: int,
    device: torch.device,
    scene_bins: bool = False,
    direct_bins: bool = False,
) -> dict[str, float | int]:
    """Evaluate full raw coherent metrics without ever exposing a test role."""

    allowed = views.role_ids(role)
    ids = tuple(allowed if source_ids is None else (int(item) for item in source_ids))
    if not ids:
        raise ValueError("evaluation role must contain at least one view")
    if any(source_id not in set(allowed) for source_id in ids):
        raise PermissionError("evaluation IDs must be part of their declared development role")

    # The field is view independent: evaluate it once, then render each view.
    field = evaluate_neural_field_tiled(model, points_m, neural_point_tile=neural_point_tile)
    weights = scale_field_to_renderer_weights(
        field,
        cell_volume_m3=cell_volume_m3,
        initial_output_scale=initial_output_scale,
    )
    numerator = 0.0
    denominator = 0.0
    objective_sum = 0.0
    partitions: dict[str, float] = {}
    for source_id in ids:
        observed, rx_pos, tx_pos = views.tensor_view(source_id, device=device)
        predicted = range_forward_operator(
            frequencies_hz,
            kvector,
            rx_pos,
            tx_pos,
            points_m,
            weights,
            phase_sign=SPINR_STYLE_PHASE_SIGN,
            pair_chunk=pair_tile,
            point_chunk=renderer_point_tile,
            compute_dtype=torch.float64,
            range_model=SPINR_STYLE_RANGE_MODEL,
        )
        residual = predicted - observed
        numerator += float(residual.abs().square().sum().item())
        denominator += float(observed.abs().square().sum().item())
        mask = (scene_range_bin_mask(frequencies_hz, rx_pos, tx_pos,
                                    support_m=SPINR_STYLE_SUPPORT_M) if scene_bins else None)
        if direct_bins:
            if mask is None:
                raise ValueError("direct-bin evaluation requires scene selection")
            bins = render_selected_bins(
                field=field, frequencies_hz=frequencies_hz, rx_pos_m=rx_pos, tx_pos_m=tx_pos,
                points_m=points_m, cell_volume_m3=cell_volume_m3,
                initial_output_scale=initial_output_scale, mask=mask,
                point_tile=renderer_point_tile, pair_tile=pair_tile)
            objective_sum += float(selected_bin_objective(
                bins, observed, mask, training_mean_raw_power=training_mean_raw_power))
        else:
            objective_sum += float(spinr_style_objective(
                predicted, observed, training_mean_raw_power=training_mean_raw_power,
                range_bin_mask=mask).item())
        if mask is not None:
            terms = spectral_partition_terms(predicted, observed, mask,
                                             training_mean_raw_power=training_mean_raw_power)
            for key, value in terms.items():
                partitions[key] = partitions.get(key, 0.0) + value
    metrics = _native_metrics(
        numerator=numerator,
        denominator=denominator,
        native_objective_sum=objective_sum,
        count=len(ids),
    )
    if scene_bins:
        for key, value in partitions.items():
            metrics[key] = value / len(ids) if key.endswith("_objective") else value
        metrics["full_spectral_objective"] = metrics["scene_objective"] + metrics["remainder_objective"]
        metrics["scene_target_energy_fraction"] = partitions["scene_target_energy"] / (
            partitions["scene_target_energy"] + partitions["remainder_target_energy"])
    return metrics


def logical_batch_update(
    *,
    model: SpinrStyleINR,
    optimizer: Adam,
    source_ids: Sequence[int],
    views: SealedRawComplexViews,
    points_m: torch.Tensor,
    cell_volume_m3: float | torch.Tensor,
    initial_output_scale: float,
    frequencies_hz: torch.Tensor,
    kvector: torch.Tensor,
    training_mean_raw_power: float,
    neural_point_tile: int,
    renderer_point_tile: int,
    pair_tile: int,
    device: torch.device,
    scene_bins: bool = False,
    direct_bins: bool = False,
) -> tuple[float, float]:
    """One exact four-view averaged optimizer update through renderer adjoints."""

    if len(source_ids) != CANONICAL_VIEW_BATCH:
        raise ValueError("each SpINR-style logical update must contain exactly four views")
    if direct_bins and not scene_bins:
        raise ValueError("direct-bin training requires scene selection")
    optimizer.zero_grad(set_to_none=True)
    model.train()
    # Field values stay fixed across the complete batch.  Their response
    # cotangents are summed before a single MLP replay/update.
    field = evaluate_neural_field_tiled(model, points_m, neural_point_tile=neural_point_tile)
    weights = scale_field_to_renderer_weights(
        field,
        cell_volume_m3=cell_volume_m3,
        initial_output_scale=initial_output_scale,
    )
    field_cotangent = torch.zeros_like(field, dtype=torch.float64)
    loss_total = 0.0
    for source_id in source_ids:
        observed, rx_pos, tx_pos = views.tensor_view(source_id, device=device)
        if direct_bins:
            loss, field_gradient = direct_bin_loss_and_field_cotangent(
                field=field, observed=observed, frequencies_hz=frequencies_hz,
                rx_pos_m=rx_pos, tx_pos_m=tx_pos, points_m=points_m, cell_volume_m3=cell_volume_m3,
                initial_output_scale=initial_output_scale, training_mean_raw_power=training_mean_raw_power,
                renderer_point_tile=renderer_point_tile, pair_tile=pair_tile)
            field_cotangent.add_(field_gradient/float(CANONICAL_VIEW_BATCH))
            loss_total += float(loss)
            continue
        with torch.no_grad():
            predicted = range_forward_operator(
                frequencies_hz,
                kvector,
                rx_pos,
                tx_pos,
                points_m,
                weights,
                phase_sign=SPINR_STYLE_PHASE_SIGN,
                pair_chunk=pair_tile,
                point_chunk=renderer_point_tile,
                compute_dtype=torch.float64,
                range_model=SPINR_STYLE_RANGE_MODEL,
            )
        loss, response_grad = response_cotangent(
            predicted,
            observed,
            training_mean_raw_power=training_mean_raw_power,
            range_bin_mask=(scene_range_bin_mask(frequencies_hz, rx_pos, tx_pos,
                                                support_m=SPINR_STYLE_SUPPORT_M) if scene_bins else None),
        )
        # Average the complete per-view objectives; no scene update occurs
        # between views or tiles.
        response_grad = response_grad / float(CANONICAL_VIEW_BATCH)
        field_cotangent.add_(real_field_cotangent_from_response(
            response_cotangent_frequency=response_grad,
            frequencies_hz=frequencies_hz,
            kvector=kvector,
            rx_pos_m=rx_pos,
            tx_pos_m=tx_pos,
            points_m=points_m,
            cell_volume_m3=cell_volume_m3,
            initial_output_scale=initial_output_scale,
            renderer_point_tile=renderer_point_tile,
            pair_tile=pair_tile,
        ))
        loss_total += float(loss.item())
    replay_field_cotangent_tiled(
        model,
        points_m,
        field_cotangent,
        neural_point_tile=neural_point_tile,
    )
    for parameter in model.parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise FloatingPointError("SpINR-style MLP has a missing or non-finite parameter gradient")
    gradient_norm = float(clip_grad_norm_(model.parameters(), max_norm=1.0).item())
    optimizer.step()
    return loss_total / float(CANONICAL_VIEW_BATCH), gradient_norm


def _execution_state(
    *,
    phase: str,
    completed_updates: int,
    loss_sum: float,
    gradient_norm_sum: float,
    elapsed_seconds: float,
    clipped_update_count: int = 0,
    stop_reason: str | None = None,
) -> dict[str, Any]:
    """Build the explicit durable state machine for one logical epoch."""

    return {
        "schema": EXECUTION_SCHEMA,
        "phase": str(phase),
        "partial_epoch": {
            "completed_updates": int(completed_updates),
            "loss_sum": float(loss_sum),
            "gradient_norm_sum": float(gradient_norm_sum),
            "elapsed_seconds": float(elapsed_seconds),
            "clipped_update_count": int(clipped_update_count),
        },
        "stop_reason": stop_reason,
    }


def _copy_execution_state(execution: Mapping[str, Any]) -> dict[str, Any]:
    """Copy only the primitive execution fields that belong in a checkpoint."""

    partial = execution["partial_epoch"]
    if not isinstance(partial, Mapping):
        raise ValueError("SpINR-style execution state lacks partial accumulators")
    return _execution_state(
        phase=str(execution["phase"]),
        completed_updates=int(partial["completed_updates"]),
        loss_sum=float(partial["loss_sum"]),
        gradient_norm_sum=float(partial["gradient_norm_sum"]),
        elapsed_seconds=float(partial["elapsed_seconds"]),
        clipped_update_count=int(partial["clipped_update_count"]),
        stop_reason=execution.get("stop_reason"),
    )


def _checkpoint_state(
    *,
    model: SpinrStyleINR,
    optimizer: Adam,
    scheduler: CosineAnnealingLR,
    sealed_contract: Mapping[str, Any],
    acquisition_identity: Mapping[str, Any],
    recipe_identity: Mapping[str, Any],
    operational_settings: Mapping[str, Any],
    training_mean_raw_power: float,
    initial_output_scale: float,
    initial_scale_ids: Sequence[int],
    initial_scale_observed_energy: float,
    initial_scale_predicted_energy: float,
    epoch_index: int,
    execution: Mapping[str, Any],
    best_validation_rel_mse: float,
    best_epoch: int | None,
    history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    execution_copy = _copy_execution_state(execution)
    partial = execution_copy["partial_epoch"]
    state = {
        "format": CHECKPOINT_FORMAT,
        "epoch_index": int(epoch_index),
        "next_update_index": int(partial["completed_updates"]),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "rng_state": capture_rng_state(),
        "sealed_npz_protocol_contract": dict(sealed_contract),
        "acquisition_identity": dict(acquisition_identity),
        "spinr_style_recipe": dict(recipe_identity),
        "operational_settings": dict(operational_settings),
        "normalization": {
            "training_mean_raw_power": float(training_mean_raw_power),
            "initial_output_scale": float(initial_output_scale),
            "initial_scale_training_ids": [int(item) for item in initial_scale_ids],
            "initial_scale_observed_energy": float(initial_scale_observed_energy),
            "initial_scale_predicted_energy": float(initial_scale_predicted_energy),
        },
        "selection": {
            "metric": "full_validation_coherent_relative_mse",
            "best_validation_rel_mse": float(best_validation_rel_mse),
            "best_epoch": None if best_epoch is None else int(best_epoch),
            "tie_policy": "earliest",
        },
        "execution": execution_copy,
        "history": [dict(item) for item in history],
    }
    if recipe_identity.get("recipe_id") in (PAPER_RECIPE, DIRECT_RECIPE, BUDGET48_RECIPE, BUDGET150_RECIPE):
        state["optimization_coverage"] = optimization_coverage(
            sealed_contract["role_ids"]["train"], epoch_index, int(partial["completed_updates"]))
    return state


def _restore_checkpoint(
    *,
    checkpoint: Mapping[str, Any],
    model: SpinrStyleINR,
    optimizer: Adam,
    scheduler: CosineAnnealingLR,
    sealed_contract: Mapping[str, Any],
    acquisition_identity: Mapping[str, Any],
    recipe_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Restore a fully prevalidated continuation without permissive fallbacks."""

    _validate_resume_checkpoint_structure(checkpoint, recipe_identity=recipe_identity)
    _validate_saved_sealed_npz_protocol_contract(
        checkpoint.get("sealed_npz_protocol_contract"), sealed_contract)
    validate_spinr_style_acquisition_identity(
        checkpoint.get("acquisition_identity"), acquisition_identity)
    normalization = checkpoint["normalization"]
    selection = checkpoint["selection"]
    if not isinstance(normalization, Mapping) or not isinstance(selection, Mapping):
        raise AssertionError("structural checkpoint validation unexpectedly did not retain mappings")
    initial_ids = tuple(int(item) for item in normalization["initial_scale_training_ids"])
    train_ids = tuple(int(item) for item in sealed_contract["role_ids"]["train"])
    if initial_ids != train_ids[:CANONICAL_INIT_SCALE_COUNT]:
        raise ValueError("SpINR-style resume would change the fixed initial-scale training rows")
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        restore_rng_state(checkpoint["rng_state"], require_complete=True)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("SpINR-style checkpoint could not restore its complete model/optimizer/RNG state") from exc
    execution = checkpoint["execution"]
    if not isinstance(execution, Mapping):
        raise AssertionError("structural checkpoint validation unexpectedly did not retain execution state")
    return {
        "epoch_index": int(checkpoint["epoch_index"]),
        "training_mean_raw_power": _finite_positive(
            normalization["training_mean_raw_power"], "saved training mean raw power"),
        "initial_scale_ids": initial_ids,
        "initial_output_scale": _finite_positive(
            normalization["initial_output_scale"], "saved initial output scale"),
        "initial_scale_observed_energy": _finite_positive(
            normalization["initial_scale_observed_energy"], "saved observed scale energy"),
        "initial_scale_predicted_energy": _finite_positive(
            normalization["initial_scale_predicted_energy"], "saved predicted scale energy"),
        "best_validation_rel_mse": _finite_nonnegative_or_infinity(
            selection["best_validation_rel_mse"], "saved best validation RelMSE"),
        "best_epoch": selection["best_epoch"],
        "history": [dict(item) for item in checkpoint["history"]],
        "execution": _copy_execution_state(execution),
    }


class _StopAfterCurrentUpdate:
    """Record TERM/INT and checkpoint after, never during, a logical update."""

    requested = False

    def __call__(self, signum: int, _frame: Any) -> None:
        self.requested = True
        print(f"Received signal {signum}; will checkpoint after the current logical update.", flush=True)


def _save_checkpoint(
    *,
    checkpoint_dir: Path,
    checkpoint_name: str,
    state: Mapping[str, Any],
) -> None:
    _atomic_torch_save(state, checkpoint_dir / checkpoint_name)


def _save_latest_and_history(
    *,
    checkpoint_dir: Path,
    state: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
) -> None:
    """Commit the authoritative latest state, then its human-readable projection.

    The selected and milestone checkpoint files are intentionally written with
    :func:`_save_checkpoint` only.  They must never advance the shared metrics
    projection ahead of ``checkpoint_latest``.
    """

    _atomic_torch_save(state, checkpoint_dir / "checkpoint_latest.pth.tar")
    _atomic_json_save({"history": [dict(item) for item in history]}, checkpoint_dir / "metrics_history.json")


def _best_recorded_metric(history: Sequence[Mapping[str, Any]], *, section: str) -> float:
    values = []
    for record in history:
        metrics = record.get(section)
        if isinstance(metrics, Mapping) and "coherent_relative_mse" in metrics:
            value = float(metrics["coherent_relative_mse"])
            if math.isfinite(value) and value >= 0:
                values.append(value)
    return min(values) if values else math.inf


def plateau_reached(history: Sequence[Mapping[str, Any]]) -> bool:
    """Implement the preregistered three 10-epoch-window plateau rule."""

    if not history:
        return False
    current_epoch = history[-1].get("epoch")
    if (isinstance(current_epoch, bool) or not isinstance(current_epoch, int)
            or current_epoch < CANONICAL_MIN_EPOCHS or current_epoch % 10 != 0):
        return False
    by_epoch = {record.get("epoch"): record for record in history if isinstance(record, Mapping)}
    anchor_epochs = tuple(current_epoch - offset for offset in (30, 20, 10, 0))
    anchors = [by_epoch.get(epoch) for epoch in anchor_epochs]
    if any(record is None for record in anchors):
        return False
    for key in ("running_best_train_diagnostic_rel_mse", "running_best_validation_rel_mse"):
        try:
            values = [float(record[key]) for record in anchors]  # type: ignore[index]
        except (KeyError, TypeError, ValueError):
            return False
        if not all(math.isfinite(value) and value >= 0 for value in values):
            return False
        for previous, current in zip(values, values[1:]):
            improvement = (previous - current) / max(abs(previous), 1e-12)
            if improvement >= 0.01:
                return False
    return True


def _epoch_budget_stop_reason(epochs: int, recipe: str = "legacy-midpoint") -> str:
    return (
        "development_epoch_budget_reached"
        if int(epochs) < (PAPER_EPOCHS if recipe in ("paper-v1-direct", "budget48-direct-1500") else CANONICAL_MIN_EPOCHS)
        else "epoch_budget_reached"
    )


def run(args: argparse.Namespace) -> None:
    _validate_cli_recipe(args)
    _disable_tf32()
    device = torch.device(args.device)
    recipe = getattr(args, "recipe", "legacy-midpoint")
    from rift.rift_dataset import collection_manifest
    num_train = (json.loads(Path(args.npz_role_manifest).read_text())["split"]["num_train"]
                 if collection_manifest(args.npz_role_manifest) else CANONICAL_TRAIN_COUNT)
    acquisition = (json.loads(Path(args.npz_role_manifest).read_text()).get('antenna_selection')
                   if collection_manifest(args.npz_role_manifest) else None)
    recipe_identity = _recipe_identity(recipe, num_train, acquisition)
    updates_per_epoch = recipe_identity["batching"]["updates_per_epoch"]
    scene_bins = recipe != "legacy-midpoint"
    direct_bins = recipe in ("paper-v1-direct", "budget48-direct", "budget48-direct-1500")

    # Resume recipe semantics and namespace protection are intentionally before
    # even the metadata-only archive loader.  A changed grid/recipe or a target
    # collision must never get as far as a raw response reader.
    resume_checkpoint = _load_resume_preflight(
        args.resume,
        recipe_identity=recipe_identity,
        requested_epochs=args.epochs,
    )
    checkpoint_dir = _resolve_output_namespace(args, resume_checkpoint)
    arrays, sealed_contract, acquisition_identity = preflight_b787_development_inputs(
        npz_path=args.npz_path,
        manifest_path=args.npz_role_manifest,
        resume_checkpoint=resume_checkpoint,
    )
    args.pair_tile = min(args.pair_tile, sealed_contract["response_shape"][1] * sealed_contract["response_shape"][2])
    recoverable_successors = _validate_checkpoint_namespace_artifacts(
        checkpoint_dir=checkpoint_dir,
        resume_checkpoint=resume_checkpoint,
        sealed_contract=sealed_contract,
        acquisition_identity=acquisition_identity,
        recipe_identity=recipe_identity,
    )
    views = materialize_b787_development_views(arrays, sealed_contract)
    print(
        f"Authorized raw-complex cache: {views.materialized_response_bytes / (1024.0 ** 3):.2f} GiB "
        "before Python/container overhead; sealed test and unused rows were not materialized.",
        flush=True,
    )
    memory_report = _enforce_memory_gates(device=device, host_rss_limit_gib=args.host_rss_limit_gib)
    print("Memory gate after authorized ingest: " + ", ".join(
        f"{key}={value:.2f}" for key, value in sorted(memory_report.items())), flush=True)
    _set_seed(CANONICAL_SEED)
    model = SpinrStyleINR().to(device=device, dtype=torch.float32)
    if model.trainable_parameter_count() != SPINR_STYLE_PARAMETER_COUNT:
        raise AssertionError("unexpected SpINR-style parameter count")
    optimizer = Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=recipe_identity["optimizer"]["cosine_max_epochs"],
        eta_min=1e-5,
    )
    if recipe in ("paper-v1", "paper-v1-direct"):
        points_m, cell_volume_m3 = gauss_legendre_cell_grid(
            args.grid_size, nodes_per_cell=NODES_PER_CELL, device=device, dtype=torch.float64)
    else:
        points_m, cell_volume_m3 = midpoint_grid(args.grid_size, device=device, dtype=torch.float64)
    print(f"Independent SpINR recipe={recipe}; integration points={len(points_m)}; "
          f"spectral bins={'scene bounds' if scene_bins else 'all'}; "
          "full-complex validation remains the selection metric.", flush=True)
    frequencies_hz = torch.as_tensor(views.frequencies_hz, device=device, dtype=torch.float64)
    kvector = get_kvector(frequencies_hz, cc).to(dtype=torch.float64)
    if frequencies_hz.shape != (600,) or not torch.isfinite(frequencies_hz).all():
        raise AssertionError("B787 metadata did not produce the required finite 600-bin frequency grid")

    history: list[dict[str, Any]]
    best_validation_rel_mse = math.inf
    best_train_diagnostic_rel_mse = math.inf
    best_epoch: int | None = None
    if resume_checkpoint is None:
        training_mean_raw_power = views.raw_training_mean_power()
        initial_output_scale, initial_scale_ids, initial_scale_observed_energy, initial_scale_predicted_energy = (
            estimate_initial_output_scale(
                model=model,
                views=views,
                points_m=points_m,
                cell_volume_m3=cell_volume_m3,
                frequencies_hz=frequencies_hz,
                kvector=kvector,
                neural_point_tile=args.neural_point_tile,
                renderer_point_tile=args.renderer_point_tile,
                pair_tile=args.pair_tile,
                device=device,
            )
        )
        epoch_index = 0
        history = []
        execution = _execution_state(
            phase="updates",
            completed_updates=0,
            loss_sum=0.0,
            gradient_norm_sum=0.0,
            elapsed_seconds=0.0,
            clipped_update_count=0,
        )
        print(
            f"Fresh fixed scale: a_init={initial_output_scale:.6e}; "
            f"raw train mean power={training_mean_raw_power:.6e}; "
            f"initial 32-view observed/random energies="
            f"{initial_scale_observed_energy:.6e}/{initial_scale_predicted_energy:.6e}",
            flush=True,
        )
    else:
        restored = _restore_checkpoint(
            checkpoint=resume_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            sealed_contract=sealed_contract,
            acquisition_identity=acquisition_identity,
            recipe_identity=recipe_identity,
        )
        epoch_index = int(restored["epoch_index"])
        training_mean_raw_power = float(restored["training_mean_raw_power"])
        initial_scale_ids = tuple(int(item) for item in restored["initial_scale_ids"])
        initial_output_scale = float(restored["initial_output_scale"])
        initial_scale_observed_energy = float(restored["initial_scale_observed_energy"])
        initial_scale_predicted_energy = float(restored["initial_scale_predicted_energy"])
        best_validation_rel_mse = float(restored["best_validation_rel_mse"])
        restored_best_epoch = restored["best_epoch"]
        best_epoch = None if restored_best_epoch is None else int(restored_best_epoch)
        history = [dict(item) for item in restored["history"]]
        raw_execution = restored["execution"]
        if not isinstance(raw_execution, Mapping):
            raise AssertionError("restored execution state must be a mapping")
        execution = _copy_execution_state(raw_execution)
        if execution["phase"] == "completed":
            raise ValueError("SpINR-style completed runs are terminal and cannot be resumed")
        best_train_diagnostic_rel_mse = _best_recorded_metric(history, section="train_diagnostic")
        print(
            f"Resuming execution phase={execution['phase']} epoch={epoch_index} "
            f"update={execution['partial_epoch']['completed_updates']}; "
            f"fixed a_init={initial_output_scale:.6e}",
            flush=True,
        )

    train_ids = views.role_ids("train")

    def state_for(
        *,
        state_epoch_index: int,
        state_execution: Mapping[str, Any],
        state_history: Sequence[Mapping[str, Any]],
        state_best_validation_rel_mse: float,
        state_best_epoch: int | None,
    ) -> dict[str, Any]:
        return _checkpoint_state(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            sealed_contract=sealed_contract,
            acquisition_identity=acquisition_identity,
            recipe_identity=recipe_identity,
            operational_settings=_operational_settings(args),
            training_mean_raw_power=training_mean_raw_power,
            initial_output_scale=initial_output_scale,
            initial_scale_ids=initial_scale_ids,
            initial_scale_observed_energy=initial_scale_observed_energy,
            initial_scale_predicted_energy=initial_scale_predicted_energy,
            epoch_index=state_epoch_index,
            execution=state_execution,
            best_validation_rel_mse=state_best_validation_rel_mse,
            best_epoch=state_best_epoch,
            history=state_history,
        )

    stopper = _StopAfterCurrentUpdate()
    original_term = signal.signal(signal.SIGTERM, stopper)
    original_int = signal.signal(signal.SIGINT, stopper)
    terminal_stop_reason: str | None = None
    try:
        while True:
            phase = execution["phase"]
            partial_epoch = execution["partial_epoch"]
            if not isinstance(partial_epoch, Mapping):
                raise AssertionError("validated execution state lost its partial epoch mapping")
            if phase == "completed":
                raise AssertionError("completed execution state reached a resumable loop")

            if phase == "pending_epoch_finalization":
                # The durable pending checkpoint was taken immediately after
                # the epoch's last update and before this scheduler step. Re-entering this
                # block after a clean interruption therefore cannot double-step
                # the scheduler or skip validation/selection.
                epoch_learning_rate = float(optimizer.param_groups[0]["lr"])
                scheduler.step()
                epoch_index += 1
                completed_updates = int(partial_epoch["completed_updates"])
                if completed_updates != updates_per_epoch:
                    raise AssertionError("pending finalization lost a completed training epoch")
                record: dict[str, Any] = {
                    "epoch": epoch_index,
                    "streamed_train_native_objective": float(partial_epoch["loss_sum"]) / completed_updates,
                    "mean_unclipped_gradient_norm": float(partial_epoch["gradient_norm_sum"]) / completed_updates,
                    "fraction_clipped_updates": (
                        int(partial_epoch["clipped_update_count"]) / completed_updates),
                    "learning_rate": epoch_learning_rate,
                    "epoch_seconds": float(partial_epoch["elapsed_seconds"]),
                }
                if scene_bins:
                    record["optimization_coverage"] = optimization_coverage(
                        train_ids, epoch_index, 0, include_per_view=False)
                should_validate = (
                    epoch_index % CANONICAL_VALIDATION_EVERY == 0 or epoch_index == args.epochs)
                is_new_best = False
                if should_validate:
                    model.eval()
                    train_metrics = evaluate_role(
                        model=model,
                        views=views,
                        role="train",
                        source_ids=train_ids[:CANONICAL_TRAIN_DIAGNOSTIC_COUNT],
                        points_m=points_m,
                        cell_volume_m3=cell_volume_m3,
                        initial_output_scale=initial_output_scale,
                        frequencies_hz=frequencies_hz,
                        kvector=kvector,
                        training_mean_raw_power=training_mean_raw_power,
                        neural_point_tile=args.neural_point_tile,
                        renderer_point_tile=args.renderer_point_tile,
                        pair_tile=args.pair_tile,
                        device=device,
                        scene_bins=scene_bins,
                        **({"direct_bins": True} if direct_bins else {}),
                    )
                    validation_metrics = evaluate_role(
                        model=model,
                        views=views,
                        role="validation",
                        source_ids=None,
                        points_m=points_m,
                        cell_volume_m3=cell_volume_m3,
                        initial_output_scale=initial_output_scale,
                        frequencies_hz=frequencies_hz,
                        kvector=kvector,
                        training_mean_raw_power=training_mean_raw_power,
                        neural_point_tile=args.neural_point_tile,
                        renderer_point_tile=args.renderer_point_tile,
                        pair_tile=args.pair_tile,
                        device=device,
                        scene_bins=scene_bins,
                        **({"direct_bins": True} if direct_bins else {}),
                    )
                    record["train_diagnostic"] = train_metrics
                    record["validation"] = validation_metrics
                    val_rel_mse = float(validation_metrics["coherent_relative_mse"])
                    best_train_diagnostic_rel_mse = min(
                        best_train_diagnostic_rel_mse,
                        float(train_metrics["coherent_relative_mse"]),
                    )
                    is_new_best = val_rel_mse < best_validation_rel_mse
                    if is_new_best:
                        best_validation_rel_mse = val_rel_mse
                        best_epoch = epoch_index
                    record["running_best_train_diagnostic_rel_mse"] = best_train_diagnostic_rel_mse
                    record["running_best_validation_rel_mse"] = best_validation_rel_mse

                committed_execution = _execution_state(
                    phase="updates",
                    completed_updates=0,
                    loss_sum=0.0,
                    gradient_norm_sum=0.0,
                    elapsed_seconds=0.0,
                    clipped_update_count=0,
                )
                candidate_history = history + [record]
                committed_candidate_state = state_for(
                    state_epoch_index=epoch_index,
                    state_execution=committed_execution,
                    state_history=candidate_history,
                    state_best_validation_rel_mse=best_validation_rel_mse,
                    state_best_epoch=best_epoch,
                )
                # A selected/milestone artifact may have been atomically saved
                # after this pending epoch's evaluation but before latest.  It
                # remains nonauthoritative until the recovered finalization
                # agrees with it.  Without this check, an altered future best
                # could overwrite the previous best and survive a replay that
                # did not actually select it.
                for artifact_name, artifact in recoverable_successors.items():
                    should_exist = (
                        (artifact_name == "checkpoint_best.pth.tar" and is_new_best)
                        or (artifact_name == "checkpoint_epoch_150.pth.tar"
                            and epoch_index == CANONICAL_MIN_EPOCHS)
                    )
                    if not should_exist or not _replayed_pending_successor_matches(
                            artifact, committed_candidate_state):
                        raise ValueError(
                            f"SpINR-style {artifact_name} disagrees with recovered pending finalization; "
                            "refusing to retain mixed checkpoint evidence")
                # The only candidate window has now been either reconciled or
                # rejected.  It must not be compared with later epochs.
                recoverable_successors.clear()
                if is_new_best:
                    _save_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        checkpoint_name="checkpoint_best.pth.tar",
                        state=committed_candidate_state,
                    )
                    print(
                        f"New validation-selected checkpoint at epoch {epoch_index}: "
                        f"coherent RelMSE={best_validation_rel_mse:.6e}",
                        flush=True,
                    )
                if epoch_index == CANONICAL_MIN_EPOCHS:
                    _save_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        checkpoint_name="checkpoint_epoch_150.pth.tar",
                        state=committed_candidate_state,
                    )
                history = candidate_history
                execution = committed_execution
                _save_latest_and_history(
                    checkpoint_dir=checkpoint_dir,
                    state=committed_candidate_state,
                    history=history,
                )
                _enforce_memory_gates(device=device, host_rss_limit_gib=args.host_rss_limit_gib)
                print(
                    f"Epoch {epoch_index}/{args.epochs}: native train="
                    f"{record['streamed_train_native_objective']:.6e}; lr={record['learning_rate']:.3e}; "
                    f"seconds={record['epoch_seconds']:.1f}",
                    flush=True,
                )
                if not direct_bins and plateau_reached(history):
                    terminal_stop_reason = "plateau_three_consecutive_10_epoch_windows"
                    print(
                        "Stopping at the registered three consecutive 10-epoch-window plateau.",
                        flush=True,
                    )
                    break
                if epoch_index >= args.epochs:
                    terminal_stop_reason = _epoch_budget_stop_reason(args.epochs, recipe)
                    break
                if stopper.requested:
                    print(
                        "Completed pending epoch finalization and saved a manager-safe recovery checkpoint.",
                        flush=True,
                    )
                    return
                continue

            if phase != "updates":
                raise AssertionError(f"unhandled SpINR-style execution phase {phase!r}")
            next_update_index = int(partial_epoch["completed_updates"])
            # A hard process loss can occur immediately after the committed
            # latest/history pair and before the normal final-state write.
            # Re-evaluate terminal policy before admitting even one new update
            # on recovery, so a qualifying epoch-150 plateau or epoch budget
            # cannot turn into an accidental epoch 151/301.
            if next_update_index == 0 and not direct_bins and plateau_reached(history):
                terminal_stop_reason = "plateau_three_consecutive_10_epoch_windows"
                break
            if next_update_index == 0 and epoch_index >= args.epochs:
                terminal_stop_reason = _epoch_budget_stop_reason(args.epochs, recipe)
                break
            loss_sum = float(partial_epoch["loss_sum"])
            grad_norm_sum = float(partial_epoch["gradient_norm_sum"])
            prior_elapsed_seconds = float(partial_epoch["elapsed_seconds"])
            clipped_update_count = int(partial_epoch["clipped_update_count"])
            batches = epoch_view_batches(train_ids, epoch=epoch_index)
            segment_started_at = time.monotonic()
            for update_index in range(next_update_index, updates_per_epoch):
                loss, grad_norm = logical_batch_update(
                    model=model,
                    optimizer=optimizer,
                    source_ids=batches[update_index],
                    views=views,
                    points_m=points_m,
                    cell_volume_m3=cell_volume_m3,
                    initial_output_scale=initial_output_scale,
                    frequencies_hz=frequencies_hz,
                    kvector=kvector,
                    training_mean_raw_power=training_mean_raw_power,
                    neural_point_tile=args.neural_point_tile,
                    renderer_point_tile=args.renderer_point_tile,
                    pair_tile=args.pair_tile,
                    device=device,
                    scene_bins=scene_bins,
                    **({"direct_bins": True} if direct_bins else {}),
                )
                loss_sum += loss
                grad_norm_sum += grad_norm
                clipped_update_count += int(grad_norm > 1.0)
                next_update_index = update_index + 1
                elapsed_seconds = prior_elapsed_seconds + (time.monotonic() - segment_started_at)
                if next_update_index == updates_per_epoch:
                    # This durable snapshot is deliberately before
                    # ``scheduler.step()``.  A TERM during validation can
                    # restart here and replay finalization exactly once.
                    execution = _execution_state(
                        phase="pending_epoch_finalization",
                        completed_updates=next_update_index,
                        loss_sum=loss_sum,
                        gradient_norm_sum=grad_norm_sum,
                        elapsed_seconds=elapsed_seconds,
                        clipped_update_count=clipped_update_count,
                    )
                    _save_latest_and_history(
                        checkpoint_dir=checkpoint_dir,
                        state=state_for(
                            state_epoch_index=epoch_index,
                            state_execution=execution,
                            state_history=history,
                            state_best_validation_rel_mse=best_validation_rel_mse,
                            state_best_epoch=best_epoch,
                        ),
                        history=history,
                    )
                    print(
                        f"Epoch {epoch_index + 1} update {updates_per_epoch} is durable; finalization is pending.",
                        flush=True,
                    )
                    break
                if stopper.requested:
                    execution = _execution_state(
                        phase="updates",
                        completed_updates=next_update_index,
                        loss_sum=loss_sum,
                        gradient_norm_sum=grad_norm_sum,
                        elapsed_seconds=elapsed_seconds,
                        clipped_update_count=clipped_update_count,
                    )
                    _save_latest_and_history(
                        checkpoint_dir=checkpoint_dir,
                        state=state_for(
                            state_epoch_index=epoch_index,
                            state_execution=execution,
                            state_history=history,
                            state_best_validation_rel_mse=best_validation_rel_mse,
                            state_best_epoch=best_epoch,
                        ),
                        history=history,
                    )
                    print("Saved a complete-update recovery checkpoint; exiting for manager-safe resume.", flush=True)
                    return
            else:
                raise AssertionError("updates phase ended before its complete training-epoch boundary")
            # The only non-terminal exit from the update loop is the durable
            # pre-scheduler pending-finalization state above.
            continue

        if terminal_stop_reason is None:
            terminal_stop_reason = _epoch_budget_stop_reason(args.epochs, recipe)
        completed_execution = _execution_state(
            phase="completed",
            completed_updates=0,
            loss_sum=0.0,
            gradient_norm_sum=0.0,
            elapsed_seconds=0.0,
            clipped_update_count=0,
            stop_reason=terminal_stop_reason,
        )
        _save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            checkpoint_name="checkpoint_final.pth.tar",
            state=state_for(
                state_epoch_index=epoch_index,
                state_execution=completed_execution,
                state_history=history,
                state_best_validation_rel_mse=best_validation_rel_mse,
                state_best_epoch=best_epoch,
            ),
        )
        phase = ("development" if _epoch_budget_stop_reason(args.epochs, recipe).startswith("development")
                 else "paper-budget" if direct_bins else "matched-exposure-or-longer")
        print(
            f"Completed {phase} run ({terminal_stop_reason}); no sealed test was opened. "
            f"Selected epoch={best_epoch}, full-validation coherent RelMSE={best_validation_rel_mse:.6e}",
            flush=True,
        )
    finally:
        signal.signal(signal.SIGTERM, original_term)
        signal.signal(signal.SIGINT, original_int)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--npz-path", default=DEFAULT_B787_NPZ_PATH,
                        help="Canonical B787 sphere10k archive on PACE; never a local data/ fallback.")
    parser.add_argument("--npz-role-manifest", default=DEFAULT_B787_MANIFEST_PATH,
                        help="Canonical sealed B787 3200/1000/1000 manifest on PACE.")
    parser.add_argument("--checkpoint-root", default="./training_checkpoints")
    parser.add_argument("--checkpoint-name", required=True,
                        help="New isolated artifact identity; never reuse a legacy sp0/RIFT checkpoint directory.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Recipe default: budget48-direct 150, historical direct 1500, historical FFT 300.")
    parser.add_argument("--recipe", choices=("legacy-midpoint", "paper-v1", "paper-v1-direct", "budget48-direct", "budget48-direct-1500"), default="legacy-midpoint",
                        help="budget48-direct: G48 midpoint/direct bins/150 epochs; budget48-direct-1500 preserves its historical schedule.")
    parser.add_argument("--grid-size", type=int, default=None,
                        help="Recipe-bound grid: 48 for budget48-direct/legacy-midpoint; 96/GL2 for historical paper-v1 recipes.")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--allow-cpu-validation", action="store_true",
                        help="Acknowledge that CPU is only for focused validation, not B787 training.")
    parser.add_argument("--neural-point-tile", type=int, default=4096)
    parser.add_argument("--renderer-point-tile", type=int, default=65536)
    parser.add_argument("--pair-tile", type=int, default=16)
    parser.add_argument("--host-rss-limit-gib", type=float, required=True,
                        help="RAM requested for the allocated validation/training job; peak RSS must stay below 80%%.")
    parser.add_argument("--resume", default=None,
                        help="Resume this exact recipe after a manager-safe interruption checkpoint.")
    args = parser.parse_args(argv)
    if args.grid_size is None:
        args.grid_size = 48 if args.recipe in ("legacy-midpoint", "budget48-direct", "budget48-direct-1500") else PARENT_GRID
    if args.epochs is None:
        args.epochs = _recipe_identity(args.recipe)["optimizer"]["cosine_max_epochs"]
    return args


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
