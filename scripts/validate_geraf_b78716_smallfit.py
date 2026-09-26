#!/usr/bin/env python3
"""Fixture and contract checks for the bounded B787 GeRaF 16/4 small fit.

No real B787 archive or response payload is opened.  On a Torch-capable
allocated node this constructs metadata-only synthetic calibrated geometry,
then verifies the new cache and checkpoint contracts.  The actual native-MF
target preparation and real renderer remain the subsequent combined GPU-cell
test, never a login-node computation.
"""

from __future__ import annotations

import copy
import inspect
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping

import numpy as np

try:
    import torch
except ImportError:
    print("SKIP: validate_geraf_b78716_smallfit.py requires Torch; use the allocated PACE GPU cell.")
    raise SystemExit(0)


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_geraf as trainer
import train_geraf_smoke as smallfit
from rift import range_operator
from rift import serialized_range_operator
from rift.config import cc
from rift.geraf_b78716_smallfit import (
    CACHE_MANIFEST_FILENAME,
    CACHE_PROTOCOL_FILENAME,
    CACHE_RECIPE_FILENAME,
    CACHE_STATS_FILENAME,
    CACHE_SCHEMA,
    MAX_UPDATES,
    MILESTONE_UPDATES,
    NUM_TRAIN,
    NUM_VALIDATION,
    SCHEDULER_HORIZON_UPDATES,
    TARGET_GRID_SHAPE,
    B78716SmallfitWorklists,
    BoundedB78716PreparationSource,
    _cache_recipe,
    _manifest,
    _protocol,
    _stats,
    _validate_complete_subset_cache,
    _view_path,
    bounded_worklists,
    target_args,
    training_args,
)
from rift.geraf_signal_operator import (
    pairwise_range_forward_operator,
    trace_and_match_magnitude,
)
from rift.geraf_b7873200_acquisition import write_or_validate_b7873200_acquisition_record
from rift.geraf_b7873200_adapter import B787ResponseHeader
from rift.geraf_b7873200_protocol import (
    B787_3200_CACHE_VIEW_DIRECTORY,
    B787_3200_TARGET_SCHEMA,
    B787_3200_NUM_TEST,
    B787_3200_NUM_UNUSED,
    B787_3200_NUM_VIEWS,
    B787_3200_SEED,
    validate_b7873200_target_cache,
)
from rift.geraf_b7873200_source import B7873200DevelopmentSource, B7873200MetadataArrays
from rift.forward_operator import get_kvector
from rift.power_baseline_dataset import atomic_save_target, atomic_write_json
from scripts import prepare_geraf_b7873200_targets as full_preparer


class Gates:
    def __init__(self) -> None:
        self.count = 0

    def check(self, condition: bool, detail: str) -> None:
        if not condition:
            raise AssertionError(detail)
        self.count += 1
        print(f"PASS {self.count:02d}: {detail}", flush=True)


def _tmp_parent() -> Path:
    configured = os.environ.get("RIFT_VALIDATION_TMPDIR")
    parent = ROOT if configured is None else Path(configured).expanduser().resolve()
    if not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
        raise RuntimeError(f"no writable validation temporary parent: {parent}")
    return parent


def _canonical_identity() -> dict[str, object]:
    permutation = np.random.Generator(np.random.PCG64(B787_3200_SEED)).permutation(B787_3200_NUM_VIEWS)
    # Canonical production role sizes, deliberately not the smallfit counts.
    validation_start = B787_3200_NUM_VIEWS - 1000
    test_start = validation_start - B787_3200_NUM_TEST
    return {
        "schema": "rift_npz_sealed_protocol_v1",
        "version": 1,
        "data_format": "npz",
        "response_shape": [10_000, 16, 16, 1, 600],
        "response_dtype": "complex64",
        "role_manifest_name": "b78710k_interp_seed42_train3200_val1000_test1000_v1",
        "split_strategy": "fixed_tail_subsampled",
        "role_ids": {
            "train": [int(value) for value in permutation[:3200]],
            "validation": [int(value) for value in permutation[validation_start:]],
            "reserved_test": [int(value) for value in permutation[test_start:validation_start]],
            "unused": [int(value) for value in permutation[3200:test_start]],
        },
        "response_access": {
            "train_materialized": True,
            "validation_materialized": True,
            "reserved_test_materialized": False,
            "unused_materialized": False,
        },
    }


def _metadata_source(identity: Mapping[str, object]) -> B7873200DevelopmentSource:
    view_count = 10_000
    viewpoints = np.zeros((view_count, 3), dtype=np.float64)
    viewpoints[:, 0] = 10.0
    coordinate = np.linspace(-0.05, 0.05, 16, dtype=np.float64)
    tx = np.broadcast_to(viewpoints[:, None, :], (view_count, 16, 3)).copy()
    rx = np.broadcast_to(viewpoints[:, None, :], (view_count, 16, 3)).copy()
    tx[:, :, 1] += coordinate[None, :]
    rx[:, :, 2] += coordinate[None, :]
    arrays = B7873200MetadataArrays(
        path="synthetic-metadata-only.npz",
        response=B787ResponseHeader((10_000, 16, 16, 1, 600), np.dtype(np.complex64)),
        viewpoint_positions=viewpoints,
        tx_pos=tx,
        rx_pos=rx,
        metadata={
            "radar_fc_hz": 10.0e9,
            "radar_bandwidth_hz": 3.0e9,
            "num_adc_samples": 600,
            "num_chirps_cpi": 1,
        },
    )
    return B7873200DevelopmentSource(arrays=arrays, identity=copy.deepcopy(dict(identity)))


def _write_leaf(
    root: Path,
    *,
    source: B7873200DevelopmentSource,
    args: SimpleNamespace,
    target_spec: Mapping[str, object],
    index: int,
    role: str,
    magnitude: float,
) -> None:
    geometry = trainer._expected_cached_geometry(
        source.arrays.viewpoint_positions[index],
        source.arrays.tx_pos[index],
        source.arrays.rx_pos[index],
        args,
    )
    atomic_save_target(
        _view_path(root, index),
        schema=np.asarray(B787_3200_TARGET_SCHEMA),
        target_spec_json=np.asarray(full_preparer._target_spec_text(target_spec)),
        view_index=np.asarray(index, dtype=np.int64),
        role=np.asarray(role),
        geraf_mf_magnitude=np.full(TARGET_GRID_SHAPE, magnitude, dtype=np.float32),
        viewpoint_position=np.asarray(source.arrays.viewpoint_positions[index], dtype=np.float32),
        **geometry,
    )


def _write_complete_fixture(root: Path) -> tuple[B7873200DevelopmentSource, SimpleNamespace, dict[str, object]]:
    identity = _canonical_identity()
    source = _metadata_source(identity)
    args = target_args(device="cpu")
    target_spec = dict(full_preparer._target_spec(args))
    worklists = bounded_worklists(identity)
    recipe = _cache_recipe(identity, target_spec)
    atomic_write_json(root / CACHE_RECIPE_FILENAME, recipe)
    write_or_validate_b7873200_acquisition_record(root, source.arrays)
    for role, indices in (("train", worklists.train), ("validation", worklists.validation)):
        for index in indices:
            _write_leaf(
                root,
                source=source,
                args=args,
                target_spec=target_spec,
                index=index,
                role=role,
                magnitude=1.0 if role == "train" else 7.0,
            )
    atomic_write_json(root / CACHE_MANIFEST_FILENAME, _manifest(worklists))
    atomic_write_json(root / CACHE_STATS_FILENAME, _stats(1.0))
    atomic_write_json(root / CACHE_PROTOCOL_FILENAME, _protocol(identity))
    return source, args, recipe


def _expect_value_error(action: Callable[[], object], detail: str, gates: Gates) -> None:
    try:
        action()
    except (ValueError, FileNotFoundError, PermissionError):
        gates.check(True, detail)
    else:
        raise AssertionError(detail)


def _check_worklists_and_source_boundary(gates: Gates) -> None:
    identity = _canonical_identity()
    worklists = bounded_worklists(identity)
    gates.check(
        worklists.train
        == (8132, 8268, 719, 8237, 4555, 6651, 5650, 9034, 1260, 644, 9098, 3760, 34, 8086, 4873, 1538)
        and worklists.validation == (1629, 6384, 9436, 5028),
        "the bounded cache selects exact ordered parent train[:16] and validation[:4] IDs",
    )
    source = _metadata_source(identity)
    bounded = BoundedB78716PreparationSource(
        arrays=source.arrays,
        parent_identity=copy.deepcopy(identity),
        worklists=worklists,
    )
    roles = identity["role_ids"]
    assert isinstance(roles, Mapping)
    for forbidden in (
        int(roles["train"][16]),
        int(roles["validation"][4]),
        int(roles["reserved_test"][0]),
        int(roles["unused"][0]),
    ):
        _expect_value_error(
            lambda forbidden=forbidden: bounded.response_view(forbidden),
            f"the bounded preparation facade denies forbidden parent response ID {forbidden}",
            gates,
        )
    _expect_value_error(
        lambda: source.arrays.response_view(worklists.train[0]),
        "the fit-compatible metadata source exposes no raw response capability",
        gates,
    )


def _check_cache_contract(gates: Gates) -> None:
    with tempfile.TemporaryDirectory(prefix=".geraf_b78716_fixture_", dir=_tmp_parent()) as temporary:
        root = Path(temporary)
        source, args, recipe = _write_complete_fixture(root)
        cache = _validate_complete_subset_cache(cache_root=root, source=source, args=args)
        gates.check(
            cache.grid_shape == TARGET_GRID_SHAPE
            and cache.train_indices == bounded_worklists(source.identity).train
            and cache.validation_indices == bounded_worklists(source.identity).validation
            and cache.geraf_mf_magnitude_peak == 1.0,
            "the distinct cache accepts exactly 20 calibrated 8-cubed native |MF| leaves and train-only peak normalization",
        )
        gates.check(
            cache.effective_pairs_per_plane == 256
            and cache.source.arrays.num_freq == 600
            and recipe["target_spec"]["operator"]["pair_chunk"] == 32
            and recipe["target_spec"]["operator"]["point_chunk"] == 512,
            "the cache records all 16x16 pairs, all 600 frequencies, and the frozen range-NUFFT chunks",
        )
        _expect_value_error(
            lambda: validate_b7873200_target_cache(root, source.identity),
            "the unchanged production 3200/1000 validator rejects the 16/4 cache root",
            gates,
        )

        worklists = bounded_worklists(source.identity)
        validation_leaf = _view_path(root, worklists.validation[0])
        _write_leaf(
            root,
            source=source,
            args=args,
            target_spec=recipe["target_spec"],
            index=worklists.validation[0],
            role="validation",
            magnitude=99.0,
        )
        cache_after_validation_change = _validate_complete_subset_cache(
            cache_root=root, source=source, args=args
        )
        gates.check(
            cache_after_validation_change.geraf_mf_magnitude_peak == 1.0,
            "the normalizer ignores validation magnitude, including a deliberately larger validation leaf",
        )
        _write_leaf(
            root,
            source=source,
            args=args,
            target_spec=recipe["target_spec"],
            index=worklists.train[0],
            role="train",
            magnitude=2.0,
        )
        _expect_value_error(
            lambda: _validate_complete_subset_cache(cache_root=root, source=source, args=args),
            "the subset reader rejects statistics that no longer equal the train-only target peak",
            gates,
        )
        _write_leaf(
            root,
            source=source,
            args=args,
            target_spec=recipe["target_spec"],
            index=worklists.train[0],
            role="train",
            magnitude=1.0,
        )
        extra = root / B787_3200_CACHE_VIEW_DIRECTORY / "view_000001.npz"
        extra.write_bytes(b"not an authorized target")
        _expect_value_error(
            lambda: _validate_complete_subset_cache(cache_root=root, source=source, args=args),
            "the subset reader rejects a target inventory with an extra non-worklist leaf",
            gates,
        )


class TinyGeRaF(torch.nn.Module):
    """Two groups matching the production optimizer grouping for checkpoint checks."""

    def __init__(self) -> None:
        super().__init__()
        self.sdf_network = torch.nn.Linear(1, 1, bias=False)
        self.other_parameter = torch.nn.Parameter(torch.tensor(0.25, dtype=torch.float32))


def _minimal_milestone(update: int) -> dict[str, object]:
    return {
        "update": int(update),
        "fixed_train_native_mf": {"mf_magnitude_mse": 1.0},
        "held_out_validation_native_mf": {"mf_magnitude_mse": 1.0},
    }


def _fixture_resource_snapshot(
    *,
    wall_seconds: float,
    process_max_rss_bytes: int | None,
    peak_torch_allocated_bytes: int,
    peak_torch_reserved_bytes: int,
    cache_size_bytes: int,
) -> dict[str, object]:
    return {
        "wall_seconds": wall_seconds,
        "process_max_rss_bytes": process_max_rss_bytes,
        "peak_torch_allocated_bytes": peak_torch_allocated_bytes,
        "peak_torch_reserved_bytes": peak_torch_reserved_bytes,
        "gpu_total_bytes": 24 * 1024**3,
        "cache_size_bytes": cache_size_bytes,
    }


def _check_schedule_and_checkpoint(gates: Gates) -> None:
    identity = _canonical_identity()
    args = target_args(device="cpu")
    with tempfile.TemporaryDirectory(prefix=".geraf_b78716_checkpoint_", dir=_tmp_parent()) as temporary:
        root = Path(temporary)
        source, _fixture_args, _fixture_recipe = _write_complete_fixture(root)
        cache = _validate_complete_subset_cache(cache_root=root, source=source, args=args)
        run_args = training_args(device="cpu")
        trainer._validate_cli(run_args)
        model = TinyGeRaF()
        optimizer = trainer.build_geraf_optimizer(model, run_args)
        scheduler = trainer.build_geraf_scheduler(
            optimizer, run_args, horizon_steps=SCHEDULER_HORIZON_UPDATES
        )
        sampler = trainer.DeterministicViewSampler(cache.train_indices, run_args.seed)
        mask_bank = trainer.PerViewDynamicMaskBank(
            cache.train_indices,
            cache.grid_shape,
            high_threshold=run_args.mask_high_threshold,
            low_ratio=run_args.mask_low_ratio,
            low_threshold=run_args.mask_low_threshold,
            device=torch.device("cpu"),
        )
        visits = {int(index): 0 for index in cache.train_indices}
        for _ in range(MAX_UPDATES):
            index = sampler.next()
            visits[index] += 1
            optimizer.zero_grad(set_to_none=True)
            loss = model.sdf_network.weight.square().sum() + model.other_parameter.square()
            loss.backward()
            trainer._require_finite_gradients(model)
            optimizer.step()
            scheduler.step()
            trainer._require_finite_optimization_state(model, optimizer, scheduler)
        gates.check(
            all(value == 2 for value in visits.values())
            and int(scheduler.state_dict()["last_epoch"]) == MAX_UPDATES
            and int(scheduler.state_dict()["T_max"]) == SCHEDULER_HORIZON_UPDATES,
            "two sampler cycles use every selected training view twice while the cosine horizon remains 50,000",
        )

        first_resources = smallfit._merge_resource_envelope(
            None,
            _fixture_resource_snapshot(
                wall_seconds=3.0,
                process_max_rss_bytes=2 * 1024**3,
                peak_torch_allocated_bytes=3 * 1024**3,
                peak_torch_reserved_bytes=4 * 1024**3,
                cache_size_bytes=11,
            ),
        )
        merged_resources = smallfit._merge_resource_envelope(
            first_resources,
            _fixture_resource_snapshot(
                wall_seconds=5.0,
                process_max_rss_bytes=3 * 1024**3,
                peak_torch_allocated_bytes=2 * 1024**3,
                peak_torch_reserved_bytes=5 * 1024**3,
                cache_size_bytes=17,
            ),
        )
        gates.check(
            merged_resources["attempt_count"] == 2
            and merged_resources["wall_seconds"] == 8.0
            and merged_resources["current_attempt_wall_seconds"] == 5.0
            and merged_resources["process_max_rss_bytes"] == 3 * 1024**3
            and merged_resources["peak_torch_allocated_bytes"] == 3 * 1024**3
            and merged_resources["peak_torch_reserved_bytes"] == 5 * 1024**3
            and merged_resources["cache_size_bytes"] == 17,
            "clean continuations preserve cumulative wall time and monotone host/GPU/cache maxima instead of reporting only the final attempt",
        )
        _expect_value_error(
            lambda: smallfit._validate_resource_envelope(None, "fixture clean checkpoint"),
            "a resumable checkpoint without cumulative resource evidence is rejected before fitting",
            gates,
        )

        # A real checkpoint payload is serialized/reloaded, then validated
        # before any next training view could be requested.
        identity_run = smallfit._run_identity(cache, run_args)
        payload = smallfit._checkpoint_payload(
            identity=identity_run,
            cache=cache,
            phase="interrupted_clean",
            completed_updates=MAX_UPDATES,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            mask_bank=mask_bank,
            visit_counts=visits,
            milestones=[_minimal_milestone(value) for value in MILESTONE_UPDATES],
            last_train={"step": MAX_UPDATES, "view_index": cache.train_indices[-1]},
            started_unix_time=1.0,
            wall_seconds=1.0,
            resource_envelope=first_resources,
        )
        serialized = root / "fixture_checkpoint.pth.tar"
        trainer._atomic_torch_save(payload, serialized)
        loaded = torch.load(serialized, map_location="cpu", weights_only=False)
        step, restored_visits, restored_milestones, _last_train, _started = smallfit._validate_checkpoint(
            loaded, identity=identity_run, cache=cache
        )
        gates.check(
            step == MAX_UPDATES
            and restored_visits == visits
            and [row["update"] for row in restored_milestones] == list(MILESTONE_UPDATES),
            "serialized checkpoint parsing validates bounded update/visit/milestone state",
        )
        shortened = copy.deepcopy(loaded)
        shortened["scheduler_state_dict"]["T_max"] = MAX_UPDATES
        _expect_value_error(
            lambda: smallfit._validate_checkpoint(shortened, identity=identity_run, cache=cache),
            "resume rejects a checkpoint with a scheduler horizon shortened to the 32-update stop budget",
            gates,
        )
        altered_identity = copy.deepcopy(identity_run)
        altered_identity["optimization"]["stop_updates"] = 31
        _expect_value_error(
            lambda: smallfit._validate_checkpoint(loaded, identity=altered_identity, cache=cache),
            "resume rejects a changed bounded recipe identity before a training view is loaded",
            gates,
        )
        mismatched_last_view = copy.deepcopy(loaded)
        mismatched_last_view["last_train"]["view_index"] = cache.validation_indices[0]
        _expect_value_error(
            lambda: smallfit._validate_checkpoint(mismatched_last_view, identity=identity_run, cache=cache),
            "resume rejects a last-update record that names a held-out validation view",
            gates,
        )

        # The final checkpoint carries the report payload so a process loss
        # between final-checkpoint and JSON-report publication can be repaired
        # without another training update or a raw-response reader.
        checkpoint_dir = root / "fit"
        checkpoint_dir.mkdir()
        initial_path = checkpoint_dir / "checkpoint_initial.pth.tar"
        best_path = checkpoint_dir / "checkpoint_best.pth.tar"
        latest_path = checkpoint_dir / "checkpoint_latest.pth.tar"
        final_path = checkpoint_dir / "checkpoint_final.pth.tar"
        trainer._atomic_torch_save({"fixture": "initial"}, initial_path)
        trainer._atomic_torch_save({"fixture": "best"}, best_path)
        terminal_report = {
            "schema": smallfit.REPORT_SCHEMA,
            "scope": smallfit.SCOPE,
            "run_identity": identity_run,
            "resources": copy.deepcopy(merged_resources),
            "checkpoints": {
                "initial": str(initial_path),
                "best": str(best_path),
                "final": str(final_path),
                "latest": str(latest_path),
                "clean_checkpoint_recovery_supported": True,
            },
        }
        terminal_payload = smallfit._checkpoint_payload(
            identity=identity_run,
            cache=cache,
            phase="complete",
            completed_updates=MAX_UPDATES,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            mask_bank=mask_bank,
            visit_counts=visits,
            milestones=[_minimal_milestone(value) for value in MILESTONE_UPDATES],
            last_train={"step": MAX_UPDATES, "view_index": cache.train_indices[-1]},
            started_unix_time=1.0,
            wall_seconds=1.0,
            resource_envelope=merged_resources,
            terminal_report=terminal_report,
        )
        trainer._atomic_torch_save(terminal_payload, final_path)
        start_paths = smallfit._fresh_paths(checkpoint_dir, str(latest_path))
        gates.check(
            start_paths.clean_resume_checkpoint is None
            and start_paths.terminal_finalize_checkpoint == final_path,
            "an explicit resume identifies a final-without-report state as metadata-only publication recovery",
        )
        recovered = smallfit._recover_terminal_report(
            checkpoint_dir=checkpoint_dir,
            final_checkpoint=final_path,
            payload=torch.load(final_path, map_location="cpu", weights_only=False),
            identity=identity_run,
            cache=cache,
        )
        gates.check(
            recovered == terminal_report
            and latest_path.is_file()
            and (checkpoint_dir / "smallfit_report.json").is_file(),
            "terminal report recovery republishes the embedded report and complete latest checkpoint without fitting",
        )


def _exit_code(action: Callable[[], None]) -> int:
    try:
        action()
    except SystemExit as exc:
        return int(exc.code)
    return 0


def _check_terminal_entrypoint(gates: Gates) -> None:
    """Exercise main's status contract without a renderer or CUDA operation."""

    original_parse_args = smallfit.parse_args
    original_run = smallfit.run
    original_stop_requested = smallfit._STOP_REQUESTED
    original_stop_signal = smallfit._STOP_SIGNAL
    with tempfile.TemporaryDirectory(prefix=".geraf_b78716_terminal_", dir=_tmp_parent()) as temporary:
        checkpoint = Path(temporary) / "checkpoint_latest.pth.tar"
        trainer._atomic_torch_save({"fixture": "clean"}, checkpoint)
        try:
            smallfit.parse_args = lambda argv=None: SimpleNamespace()  # type: ignore[assignment]
            smallfit.run = lambda _args: {"complete": True}  # type: ignore[assignment]
            completion_status = _exit_code(lambda: smallfit.main([]))
            smallfit.run = lambda _args: smallfit._CleanStop(  # type: ignore[assignment]
                checkpoint=checkpoint, signal=None
            )
            clean_stop_status = _exit_code(lambda: smallfit.main([]))
            smallfit.run = lambda _args: smallfit._PreFitStop(signal=None)  # type: ignore[assignment]
            pre_fit_stop_status = _exit_code(lambda: smallfit.main([]))
        finally:
            smallfit.parse_args = original_parse_args  # type: ignore[assignment]
            smallfit.run = original_run  # type: ignore[assignment]
    try:
        smallfit._STOP_REQUESTED = True
        smallfit._STOP_SIGNAL = 15
        fresh_stop = smallfit._pre_fit_stop_outcome(fresh_fit=True)
        clean_resume_stop = smallfit._pre_fit_stop_outcome(fresh_fit=False)
        terminal_recovery_stop = smallfit._pre_fit_stop_outcome(fresh_fit=False)
    finally:
        smallfit._STOP_REQUESTED = original_stop_requested
        smallfit._STOP_SIGNAL = original_stop_signal
    launcher = (ROOT / "slurm" / "validate_geraf_b78716_smallfit_v1.sbatch").read_text(
        encoding="utf-8"
    )
    gates.check(
        completion_status == 0
        and clean_stop_status == smallfit.CLEAN_STOP_EXIT_CODE
        and pre_fit_stop_status == smallfit.PRE_FIT_STOP_EXIT_CODE,
        "the driver entrypoint separates successful lifecycle completion, resumable clean stop, and pre-fit stop statuses",
    )
    gates.check(
        isinstance(fresh_stop, smallfit._PreFitStop)
        and clean_resume_stop is None
        and terminal_recovery_stop is None,
        "a signal during metadata loading stops only a fresh preparation; clean resume and terminal-report recovery remain resumable",
    )
    gates.check(
        'print("GERAF_B78716_SMALLFIT_LIFECYCLE_COMPLETE", flush=True)' in inspect.getsource(smallfit.main)
        and "grep -Fxq 'GERAF_B78716_SMALLFIT_LIFECYCLE_COMPLETE'" in launcher
        and "driver_status == 143" in launcher
        and "driver_status == 75" in launcher
        and "attempt_log_dir" in launcher,
        "the launcher contract matches the driver's exact success marker, stop statuses, and durable driver logs",
    )


def _check_reuse_seams(gates: Gates) -> None:
    update_source = inspect.getsource(trainer.train_one_view_update)
    evaluator_source = inspect.getsource(trainer.evaluate_indices)
    driver_source = inspect.getsource(smallfit.run)
    preparer_source = inspect.getsource(full_preparer._prepare_one)
    gates.check(
        "mask_bank.valid_mask" in update_source
        and "masked_magnitude_l2" in update_source
        and "scheduler.step" in update_source,
        "the bounded driver reuses the extracted production single-view optimizer semantics",
    )
    gates.check(
        "mask_bank" not in evaluator_source and "role" in evaluator_source,
        "the role-index evaluator has no dynamic-mask update path for held-out validation",
    )
    gates.check(
        "prepare_complete_subset_cache" in driver_source
        and "load_complete_subset_cache" in driver_source
        and "while step < MAX_UPDATES" in driver_source,
        "one driver invokes complete target preparation before the bounded fit loop",
    )
    gates.check(
        "response = source.response_view(index)" in preparer_source
        and "(16, 16, 600)" in preparer_source
        and "_matched_filter_amplitude" in preparer_source,
        "target preparation reuses the established chirp-averaged full 16x16x600 native-MF route",
    )


def _check_inference_cache_transition(gates: Gates) -> None:
    """Exercise the exact preparation-to-autograd cache transition in one process.

    This deliberately does not clear ``_DEAPOD_CACHE`` between inference
    preparation and differentiable rendering.  The uncommon grid parameters
    make the first request cold while keeping this actual CUDA/Torch check
    tiny enough to belong in the one bounded GeRaF allocation.
    """

    if not torch.cuda.is_available():
        raise RuntimeError("the allocated GeRaF cache-transition fixture requires CUDA")
    device = torch.device("cuda:0")
    torch.manual_seed(21_067)
    nf_full = 29
    oversample = 3
    kernel_width = 6
    m_grid = range_operator._next_power_of_two(nf_full * oversample)
    tau_bins = range_operator._default_tau_bins(kernel_width)
    key = (
        nf_full,
        m_grid,
        kernel_width,
        round(float(tau_bins), 12),
        device.type,
        device.index,
        str(torch.float64),
    )
    if key in range_operator._DEAPOD_CACHE:
        raise AssertionError("cache-transition fixture requires its dedicated deapodization key to be cold")

    frequencies = torch.linspace(9.85e9, 10.15e9, nf_full, dtype=torch.float64, device=device)
    kvector = get_kvector(frequencies, cc)
    tx_positions = torch.tensor(
        ((10.0, -0.025, 0.010), (10.0, 0.025, -0.010)),
        dtype=torch.float64,
        device=device,
    )
    rx_positions = torch.tensor(
        ((10.0, -0.015, -0.020), (10.0, 0.015, 0.020)),
        dtype=torch.float64,
        device=device,
    )
    sample_positions = torch.tensor(
        ((-0.050, -0.025, 0.010), (0.020, 0.040, -0.015), (0.060, -0.030, 0.025)),
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    amplitude_re = torch.tensor(
        ((0.11, -0.08, 0.06), (0.07, 0.09, -0.05), (-0.04, 0.03, 0.10), (0.05, -0.07, 0.02)),
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    amplitude_im = torch.tensor(
        ((-0.03, 0.05, 0.08), (0.04, -0.06, 0.01), (0.09, 0.02, -0.07), (-0.05, 0.06, 0.03)),
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    pair_amplitudes = torch.complex(amplitude_re, amplitude_im)
    operator_kwargs = {
        "phase_sign": -1.0,
        "oversample": oversample,
        "kernel_width": kernel_width,
        "pair_chunk": 2,
        "point_chunk": 2,
        "compute_dtype": torch.float64,
    }

    # This is the same ordering as real target preparation: the first shared
    # deapodization request happens under inference mode.
    with torch.inference_mode():
        prepared_response = pairwise_range_forward_operator(
            frequencies,
            kvector,
            tx_positions,
            rx_positions,
            sample_positions.detach(),
            pair_amplitudes.detach(),
            **operator_kwargs,
        )
        cold_deapod = range_operator._deapodization(
            nf_full, m_grid, kernel_width, tau_bins, device, torch.float64
        )
    gates.check(
        bool(torch.isfinite(prepared_response.real).all())
        and bool(torch.isfinite(prepared_response.imag).all())
        and not cold_deapod.is_inference()
        and not cold_deapod.requires_grad,
        "inference-mode GeRaF preparation creates a finite ordinary no-grad shared deapodization constant",
    )

    response, matched_magnitude = trace_and_match_magnitude(
        frequencies,
        kvector,
        tx_positions,
        rx_positions,
        sample_positions,
        pair_amplitudes,
        sample_positions,
        **operator_kwargs,
    )
    ge_raf_loss = response.abs().square().mean() + matched_magnitude.square().mean()
    ge_raf_loss.backward()
    gates.check(
        bool(torch.isfinite(ge_raf_loss))
        and all(
            value is not None
            and bool(torch.isfinite(value).all())
            and bool(value.abs().max() > 0.0)
            for value in (sample_positions.grad, amplitude_re.grad, amplitude_im.grad)
        ),
        "a differentiable GeRaF trace-and-native-matched-filter backward pass succeeds after inference preparation",
    )

    # Simulate a stale cache entry left by pre-fix code, without clearing the
    # shared cache or changing its key/value numerics.  The helper must repair
    # this entry once and then reuse its ordinary replacement.
    with torch.inference_mode():
        stale_deapod = cold_deapod.clone()
    if not stale_deapod.is_inference():
        raise AssertionError("fixture failed to construct an inference-mode legacy cache entry")
    range_operator._DEAPOD_CACHE[key] = stale_deapod
    normalized_deapod = range_operator._deapodization(
        nf_full, m_grid, kernel_width, tau_bins, device, torch.float64
    )
    warm_deapod = range_operator._deapodization(
        nf_full, m_grid, kernel_width, tau_bins, device, torch.float64
    )
    gates.check(
        not normalized_deapod.is_inference()
        and normalized_deapod is warm_deapod
        and torch.equal(normalized_deapod, cold_deapod),
        "a pre-existing inference cache entry is converted once to the identical ordinary constant and warm reuse is allocation-free",
    )

    rift_positions = sample_positions.detach()
    rift_re = torch.tensor((0.09, -0.04, 0.06), dtype=torch.float64, device=device, requires_grad=True)
    rift_im = torch.tensor((-0.02, 0.07, -0.05), dtype=torch.float64, device=device, requires_grad=True)
    rift_weights = torch.complex(rift_re, rift_im)
    rift_kwargs = {
        "phase_sign": -1.0,
        "oversample": oversample,
        "kernel_width": kernel_width,
        "pair_chunk": 2,
        "compute_dtype": torch.float64,
        "range_model": "none",
    }
    ordinary = range_operator.range_forward_operator(
        frequencies,
        kvector,
        rx_positions,
        tx_positions,
        rift_positions,
        rift_weights,
        point_chunk=2,
        **rift_kwargs,
    )

    def rift_chunks() -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [
            (rift_positions[:2], rift_weights[:2]),
            (rift_positions[2:], rift_weights[2:]),
        ]

    serialized = serialized_range_operator.range_forward_operator_chunks(
        frequencies,
        kvector,
        rx_positions,
        tx_positions,
        rift_chunks,
        **rift_kwargs,
    )
    ordinary.abs().square().mean().backward()
    gates.check(
        torch.allclose(serialized, ordinary.detach(), rtol=1.0e-11, atol=1.0e-11)
        and all(
            value is not None and bool(torch.isfinite(value).all()) and bool(value.abs().max() > 0.0)
            for value in (rift_re.grad, rift_im.grad)
        ),
        "ordinary and serialized RIFT forward rendering agree on the repaired shared cache and retain finite ordinary-renderer gradients",
    )

    residual_re = torch.linspace(-0.08, 0.09, nf_full * 4, dtype=torch.float64, device=device).view(nf_full, 2, 2)
    residual_im = torch.linspace(0.06, -0.05, nf_full * 4, dtype=torch.float64, device=device).view(nf_full, 2, 2)
    residual = torch.complex(residual_re, residual_im)
    ordinary_adjoint = range_operator.range_adjoint_operator(
        frequencies,
        kvector,
        rx_positions,
        tx_positions,
        rift_positions,
        residual,
        point_chunk=2,
        **rift_kwargs,
    )

    def position_chunks() -> list[torch.Tensor]:
        return [rift_positions[:2], rift_positions[2:]]

    serialized_adjoint = serialized_range_operator.range_adjoint_operator_chunks(
        frequencies,
        kvector,
        rx_positions,
        tx_positions,
        position_chunks,
        residual,
        **rift_kwargs,
    )
    gates.check(
        torch.allclose(serialized_adjoint, ordinary_adjoint, rtol=1.0e-11, atol=1.0e-11)
        and bool(torch.isfinite(ordinary_adjoint.real).all())
        and bool(torch.isfinite(ordinary_adjoint.imag).all()),
        "ordinary and serialized RIFT adjoints agree after the inference-to-autograd cache transition",
    )


def main() -> None:
    gates = Gates()
    _check_worklists_and_source_boundary(gates)
    _check_cache_contract(gates)
    _check_schedule_and_checkpoint(gates)
    _check_terminal_entrypoint(gates)
    _check_reuse_seams(gates)
    print(f"GERAF_B78716_SMALLFIT_VALIDATION_PASS gates={gates.count}", flush=True)


if __name__ == "__main__":
    main()
