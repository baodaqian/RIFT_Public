"""Allocated-node, data-free engineering validation for corrected SE B7873200.

The test creates only synthetic tensors beneath a temporary directory.  It
never opens the B787 archive, role manifest, a response accessor, a baseline
checkpoint, or a real output root.  It exercises the actual Stage-1 final
bundle validator, Stage-2 checkpoint/lifecycle I/O, clean-resume permissions,
and atomic completion-package promotion.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import signal
import tempfile
import types
from typing import Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rift import sugavanam_ertin_b7873200_stage1 as stage1  # noqa: E402
from rift.sugavanam_ertin_stage2_runtime_v1 import (  # noqa: E402
    CLEAN_INTERRUPTION_EXIT_CODE,
    LifecyclePhase,
    LifecycleState,
    RuntimeContractError,
    atomic_json_dump,
    atomic_torch_save,
    clean_interruption_record,
    load_json_mapping,
    load_torch_mapping,
    prepare_run_root,
    promote_complete_package,
    same_value,
    validate_lifecycle_record,
    write_lifecycle,
)
import train_sugavanam_ertin_stage2 as trainer  # noqa: E402


def reject(label: str, fn) -> None:
    try:
        fn()
    except (ValueError, RuntimeError, FileNotFoundError, PermissionError):
        return
    raise AssertionError(f"expected rejection: {label}")


def canonical_contract() -> dict[str, object]:
    perm = np.random.Generator(np.random.PCG64(42)).permutation(10_000)
    return {
        "schema": "rift_npz_sealed_protocol_v1",
        "version": 1,
        "data_format": "npz",
        "response_shape": [10000, 16, 16, 1, 600],
        "response_dtype": "complex64",
        "role_manifest_name": stage1.B787_3200_MANIFEST_NAME,
        "split_strategy": "fixed_tail_subsampled",
        "role_ids": {
            "train": [int(item) for item in perm[:3200]],
            "unused": [int(item) for item in perm[3200:8000]],
            "reserved_test": [int(item) for item in perm[8000:9000]],
            "validation": [int(item) for item in perm[9000:]],
        },
        "response_access": {
            "train_materialized": True,
            "validation_materialized": True,
            "reserved_test_materialized": False,
            "unused_materialized": False,
        },
    }


def synthetic_acquisition(sealed: dict[str, object]) -> dict[str, object]:
    # Keep this artificial pose surface small in value range but full in shape;
    # it tests identity layout without any archive payload.
    ids = np.arange(10000, dtype=np.float64)[:, None, None]
    channels = np.arange(16, dtype=np.float64)[None, :, None]
    axes = np.arange(3, dtype=np.float64)[None, None, :]
    arrays = {
        "response": None,
        "meta": {
            "target_type": "b787",
            "experiment": "sphere10k",
            "radar_fc_hz": 10.0e9,
            "radar_bandwidth_hz": 3.0e9,
            "num_adc_samples": 600,
        },
        "tx_pos": np.ascontiguousarray(ids * 1.0e-6 + channels * 1.0e-8 + axes * 1.0e-10),
        "rx_pos": np.ascontiguousarray(-ids * 1.0e-6 - channels * 1.0e-8 - axes * 1.0e-10),
    }
    return stage1.build_b7873200_acquisition_identity(arrays, sealed)


def synthetic_stage1_final(recipe: dict[str, object]) -> dict[str, object]:
    grid = int(recipe["scene"]["granularity"])
    extent = float(recipe["scene"]["extent_m"])
    w_re = torch.zeros((grid, grid, grid), dtype=torch.float32)
    w_im = torch.zeros_like(w_re)
    active = torch.zeros((grid, grid, grid), dtype=torch.bool)
    for index, magnitude in (
        ((20, 20, 20), 1.0),
        ((23, 20, 20), 0.9),
        ((20, 23, 20), 0.8),
        ((20, 20, 23), 0.7),
    ):
        w_re[index] = magnitude
        active[index] = True
    gain_log_mag = torch.zeros((), dtype=torch.float32)
    gain_phase = torch.zeros((), dtype=torch.float32)
    completed_adam_steps = int(recipe["fit"]["epochs"]) * stage1.B787_3200_NUM_TRAIN
    state = {
        "epoch": int(recipe["fit"]["epochs"]),
        "loss": 1.0,
        "scene_repr": "grid",
        "range_model": recipe["observation"]["range_model"],
        "extent": extent,
        "granularity": grid,
        "l1_weight": float(recipe["fit"]["l1_weight"]),
        "adam_eps": float(recipe["fit"]["optimizer"]["eps"]),
        "adaptive_capacity_v2": False,
        "val_cap_axis": None,
        "sealed_npz_protocol_contract": copy.deepcopy(recipe["sealed_protocol_identity"]),
        "execution_contract": stage1.expected_generic_execution_contract(recipe),
        "gain_state_dict": {
            "log_mag": gain_log_mag,
            "phase": gain_phase,
            "initialized": torch.ones((), dtype=torch.bool),
        },
        "optimizer_state_dict": {
            "state": {
                0: {
                    "step": torch.tensor(float(completed_adam_steps)),
                    "exp_avg": torch.zeros_like(w_re),
                    "exp_avg_sq": torch.zeros_like(w_re),
                },
                1: {
                    "step": torch.tensor(float(completed_adam_steps)),
                    "exp_avg": torch.zeros_like(w_im),
                    "exp_avg_sq": torch.zeros_like(w_im),
                },
                2: {
                    "step": torch.tensor(float(completed_adam_steps)),
                    "exp_avg": torch.zeros_like(gain_log_mag),
                    "exp_avg_sq": torch.zeros_like(gain_log_mag),
                },
                3: {
                    "step": torch.tensor(float(completed_adam_steps)),
                    "exp_avg": torch.zeros_like(gain_phase),
                    "exp_avg_sq": torch.zeros_like(gain_phase),
                },
            },
            "param_groups": [
                {
                    "params": [0, 1],
                    "lr": float(recipe["fit"]["optimizer"]["lr"]),
                    "eps": float(recipe["fit"]["optimizer"]["eps"]),
                    "weight_decay": float(recipe["fit"]["optimizer"]["weight_decay"]),
                    "betas": tuple(recipe["fit"]["optimizer"]["betas"]),
                },
                {
                    "params": [2, 3],
                    "lr": float(recipe["fit"]["optimizer"]["lr"]),
                    "eps": float(recipe["fit"]["optimizer"]["eps"]),
                    "weight_decay": float(recipe["fit"]["optimizer"]["weight_decay"]),
                    "betas": tuple(recipe["fit"]["optimizer"]["betas"]),
                },
            ],
        },
        "scheduler_state_dict": {
            "T_0": int(recipe["fit"]["scheduler"]["t0"]),
            "T_mult": int(recipe["fit"]["scheduler"]["t_mult"]),
            "eta_min": float(recipe["fit"]["scheduler"]["eta_min"]),
            "last_epoch": int(recipe["fit"]["epochs"]),
            "base_lrs": [float(recipe["fit"]["optimizer"]["lr"])] * 2,
            "_last_lr": [float(recipe["fit"]["optimizer"]["lr"])] * 2,
            "T_i": 160,
            "T_cur": 0,
        },
        "model_state_dict": {
            "w_re": w_re,
            "w_im": w_im,
            "active_mask": active,
            "grid_positions": torch.from_numpy(
                stage1.expected_cell_centred_grid(grid, extent, dtype=np.dtype(np.float32))
            ),
        },
    }
    return state


def tiny_recipe() -> dict[str, object]:
    recipe = trainer.default_stage2_recipe()
    recipe.update(
        {
            "steps": 1,
            "init_steps": 2,
            "init_batch": 32,
            "batch_on": 4,
            "batch_off": 4,
            "batch_iso": 4,
            "batch_signed": 4,
            "batch_boundary": 4,
            "batch_inner": 4,
            "n_iso": 8,
            "hidden_dim": 8,
            "n_layers": 5,
            "n_fourier": 2,
            "gate_grid": 16,
            "mesh_grid": 32,
            "inner_anchor_count": 8,
            "boundary_shell_resolution": 3,
            "iso_start": 1,
            "iso_refresh": 1,
            "gate_every": 1,
            "save_every": 1,
        }
    )
    return recipe


def analytic_fixture_model(
    recipe: dict[str, object],
    geometry: dict[str, object],
    *,
    factory=None,
) -> torch.nn.Module:
    """Keep the real Fourier state layout while making a known closed SDF.

    The production validator reconstructs a normal ``FourierFeatureSDF``.
    This allocated-node fixture replaces only ``forward`` on its in-memory
    instance, so the test can exercise actual initialization, projection,
    marching-cubes export, and package validation against a deterministic
    field without claiming that a tiny two-step MLP fit is scientific.
    """

    model_factory = trainer._model if factory is None else factory
    model = model_factory(recipe, float(geometry["extent"]), torch.device("cpu"))
    center = tuple(float(item) for item in geometry["spec"].center)
    radius = float(geometry["spec"].radius)

    def analytic_forward(self, xyz: torch.Tensor) -> torch.Tensor:
        centre = xyz.new_tensor(center)
        zero = sum((parameter.sum() * 0.0 for parameter in self.parameters()), xyz.new_zeros(()))
        return (xyz - centre).norm(dim=-1) - radius + zero

    model.forward = types.MethodType(analytic_forward, model)
    return model


def history_row(step: int, gate: dict[str, object], optimizer: torch.optim.Optimizer, loss: float, iso_count: int) -> dict[str, object]:
    return {
        "step": step,
        "total": loss,
        "on": loss,
        "normal": 0.0,
        "signed": 0.0,
        "off": 0.0,
        "boundary": 0.0,
        "inner": 0.0,
        "iso": 0.0,
        "iso_normal": 0.0,
        "eik": 0.0,
        "effective_lambda_off": 0.0,
        "iso_count": iso_count,
        "gate_field_min": float(gate["field_min"]),
        "gate_boundary_min": float(gate["boundary_min"]),
        "gate_shell_min": float(gate["protected_shell"]["minimum"]),
        "lr": float(optimizer.param_groups[0]["lr"]),
        "seconds": 0.0,
    }


class SyntheticRunContract:
    """Temporary output identity used only by this allocated-node validator."""

    def __init__(self, root: Path) -> None:
        self.output_dir = str(root)

    @property
    def latest_checkpoint(self) -> str:
        return str(Path(self.output_dir) / "checkpoint_latest.pth.tar")

    @property
    def lifecycle_path(self) -> str:
        return str(Path(self.output_dir) / "lifecycle.json")

    @property
    def complete_dir(self) -> str:
        return str(Path(self.output_dir) / "complete")


def _history_without_elapsed(history: object) -> list[dict[str, object]]:
    if not isinstance(history, list):
        raise AssertionError("expected a Stage-2 history list")
    normalized: list[dict[str, object]] = []
    for row in history:
        if not isinstance(row, dict):
            raise AssertionError("expected a Stage-2 history row")
        copied = copy.deepcopy(row)
        copied.pop("seconds", None)
        normalized.append(copied)
    return normalized


def _assert_same_fields(
    left: Mapping[str, object], right: Mapping[str, object], fields: tuple[str, ...], label: str
) -> None:
    for field in fields:
        if not same_value(left.get(field), right.get(field)):
            raise AssertionError(f"{label}: changed field {field}")


def actual_entrypoint_recovery_checks(
    *,
    root: Path,
    source: Mapping[str, object],
    recipe: dict[str, object],
) -> None:
    """Exercise the sealed entrypoint's real clean-stop and resume machinery.

    The production ``_run`` remains canonical-only.  This test temporarily
    replaces its imported input/provenance seams with an already validated,
    data-free cloud, then uses the real SIGTERM handler, checkpoint I/O,
    Adam/scheduler/RNG restore, export, and atomic completion path.
    """

    contract = SyntheticRunContract(root / "recovered")
    reference_contract = SyntheticRunContract(root / "uninterrupted")
    initial_args = types.SimpleNamespace(resume=None, device="cpu")
    geometry = trainer._stage1_geometry(source, recipe, torch.device("cpu"))
    checkpoint_contract = {
        "contract": {
            "method": trainer.METHOD_NAME,
            "campaign_identity": trainer.CAMPAIGN_IDENTITY,
            "artifact_identity": trainer.ARTIFACT_IDENTITY,
            "policy_identity": trainer.POLICY_IDENTITY,
            "implementation_kind": trainer.IMPLEMENTATION_KIND,
        },
        "synthetic": "entrypoint-checkpoint-contract-v1",
    }
    lifecycle_contract = {"synthetic": "entrypoint-lifecycle-contract-v1"}

    # The unmodified public entrypoint must reject this temporary contract and
    # reduced recipe before any test-local patch makes it runnable.
    reject(
        "public entrypoint accepted synthetic contract",
        lambda: trainer._run(initial_args, contract=contract, recipe=recipe),
    )

    original_model = trainer._model
    originals = {
        "validate_contract": trainer.validate_contract,
        "validate_stage2_recipe": trainer.validate_stage2_recipe,
        "load_validated_stage1_source": trainer.load_validated_stage1_source,
        "stage2_provenance_record": trainer.stage2_provenance_record,
        "lifecycle_contract_record": trainer.lifecycle_contract_record,
        "_model": trainer._model,
    }
    original_sigterm = signal.getsignal(signal.SIGTERM)
    original_sigint = signal.getsignal(signal.SIGINT)

    trainer.validate_contract = lambda candidate: candidate
    trainer.validate_stage2_recipe = lambda candidate: copy.deepcopy(dict(candidate))
    trainer.load_validated_stage1_source = lambda _contract: copy.deepcopy(dict(source))
    trainer.stage2_provenance_record = (
        lambda _contract, _stage1_record, _recipe: copy.deepcopy(checkpoint_contract)
    )
    trainer.lifecycle_contract_record = lambda _provenance: copy.deepcopy(lifecycle_contract)
    trainer._model = lambda local_recipe, _extent, _device: analytic_fixture_model(
        local_recipe, geometry, factory=original_model
    )

    def run_with_clean_stop(
        args: types.SimpleNamespace, active_contract: SyntheticRunContract, target_check: int
    ) -> int:
        """Ask the real installed SIGTERM handler to stop at one checked boundary."""

        checks = 0
        original_stop = trainer.stop_requested

        def injecting_stop() -> bool:
            nonlocal checks
            checks += 1
            if checks == target_check:
                os.kill(os.getpid(), signal.SIGTERM)
            return original_stop()

        trainer.stop_requested = injecting_stop
        try:
            return trainer._run(args, contract=active_contract, recipe=recipe)
        finally:
            trainer.stop_requested = original_stop
            trainer.reset_stop_request()

    try:
        # Checks one and two are inside the two initialization updates.  The
        # second therefore exercises a real clean stop after the last update,
        # before the initialization gate/phase transition.
        first_exit = run_with_clean_stop(initial_args, contract, target_check=2)
        if first_exit != CLEAN_INTERRUPTION_EXIT_CODE:
            raise AssertionError(f"expected final-init clean interruption, got {first_exit}")
        first_lifecycle = load_json_mapping(contract.lifecycle_path, "final-init clean lifecycle")
        validate_lifecycle_record(
            first_lifecycle,
            expected_contract=lifecycle_contract,
            max_steps=int(recipe["steps"]),
        )
        if (
            first_lifecycle["phase"] != LifecyclePhase.INITIALIZATION.value
            or first_lifecycle["state"] != LifecycleState.CLEAN_INTERRUPTED.value
            or first_lifecycle["resume_allowed"] is not True
        ):
            raise AssertionError("final-init interruption was not recorded as a clean resumable state")
        initialization_state = load_torch_mapping(contract.latest_checkpoint, "final-init latest")
        initialization_model = trainer._model(recipe, float(geometry["extent"]), torch.device("cpu"))
        if trainer._validate_checkpoint(
            initialization_state,
            contract_record=checkpoint_contract,
            recipe=recipe,
            geometry=geometry,
            model=initialization_model,
            require_resumable=True,
        ) is not LifecyclePhase.INITIALIZATION:
            raise AssertionError("final-init latest did not retain initialization phase")
        saved_init = initialization_state["initialization"]
        if (
            initialization_state["init_step"] != int(recipe["init_steps"])
            or not isinstance(saved_init, Mapping)
            or saved_init.get("initial_gate") is not None
            or not np.isfinite(float(saved_init.get("final_loss", np.nan)))
        ):
            raise AssertionError("final-init latest did not preserve the unreplayed update boundary")

        # The resume executes the final-init gate without an optimizer replay.
        # Its second checked boundary is immediately after the real
        # EXPORT_PENDING checkpoint, with optimizer/scheduler/RNG state saved.
        resume_args = types.SimpleNamespace(resume=contract.latest_checkpoint, device="cpu")
        second_exit = run_with_clean_stop(resume_args, contract, target_check=2)
        if second_exit != CLEAN_INTERRUPTION_EXIT_CODE:
            raise AssertionError(f"expected export-pending clean interruption, got {second_exit}")
        pending_lifecycle = load_json_mapping(contract.lifecycle_path, "export-pending clean lifecycle")
        validate_lifecycle_record(
            pending_lifecycle,
            expected_contract=lifecycle_contract,
            max_steps=int(recipe["steps"]),
        )
        if (
            pending_lifecycle["phase"] != LifecyclePhase.EXPORT_PENDING.value
            or pending_lifecycle["state"] != LifecycleState.CLEAN_INTERRUPTED.value
            or pending_lifecycle["resume_allowed"] is not True
        ):
            raise AssertionError("export-pending interruption was not recorded as a clean resumable state")
        pending = load_torch_mapping(contract.latest_checkpoint, "export-pending latest")
        pending_model = trainer._model(recipe, float(geometry["extent"]), torch.device("cpu"))
        if trainer._validate_checkpoint(
            pending,
            contract_record=checkpoint_contract,
            recipe=recipe,
            geometry=geometry,
            model=pending_model,
            require_resumable=True,
        ) is not LifecyclePhase.EXPORT_PENDING:
            raise AssertionError("export-pending latest did not retain a validated export phase")
        if pending["step"] != int(recipe["steps"]) or len(pending["history"]) != int(recipe["steps"]):
            raise AssertionError("export-pending latest lacks completed training evidence")
        preserved = {
            name: copy.deepcopy(pending[name])
            for name in (
                "model_state_dict",
                "optimizer_state_dict",
                "scheduler_state_dict",
                "initialization",
                "best_loss",
                "gate_history",
                "iso_refresh_history",
                "iso_points",
                "iso_normals",
                "sample_rng_state",
                "rng_state",
            )
        }
        preserved_history = _history_without_elapsed(pending["history"])

        complete_exit = trainer._run(resume_args, contract=contract, recipe=recipe)
        if complete_exit != 0:
            raise AssertionError(f"export-pending recovery did not complete: {complete_exit}")
        recovered = load_torch_mapping(
            Path(contract.complete_dir) / trainer.FINAL_CHECKPOINT_NAME,
            "recovered final checkpoint",
        )
        _assert_same_fields(
            recovered,
            preserved,
            tuple(preserved),
            "export-pending recovery rewrote trajectory state",
        )
        if not same_value(_history_without_elapsed(recovered["history"]), preserved_history):
            raise AssertionError("export-pending recovery rewrote training history")

        reference_exit = trainer._run(initial_args, contract=reference_contract, recipe=recipe)
        if reference_exit != 0:
            raise AssertionError(f"uninterrupted reference did not complete: {reference_exit}")
        reference = load_torch_mapping(
            Path(reference_contract.complete_dir) / trainer.FINAL_CHECKPOINT_NAME,
            "uninterrupted final checkpoint",
        )
        _assert_same_fields(
            recovered,
            reference,
            (
                "model_state_dict",
                "optimizer_state_dict",
                "scheduler_state_dict",
                "initialization",
                "best_loss",
                "gate_history",
                "iso_refresh_history",
                "iso_points",
                "iso_normals",
                "sample_rng_state",
                "rng_state",
            ),
            "clean-resumed and uninterrupted trajectories differ",
        )
        if not same_value(
            _history_without_elapsed(recovered["history"]),
            _history_without_elapsed(reference["history"]),
        ):
            raise AssertionError("clean-resumed and uninterrupted histories differ")
    finally:
        for name, value in originals.items():
            setattr(trainer, name, value)
        trainer.reset_stop_request()
        signal.signal(signal.SIGTERM, original_sigterm)
        signal.signal(signal.SIGINT, original_sigint)


def main() -> None:
    sealed = stage1.canonical_sealed_identity(canonical_contract())
    acquisition = synthetic_acquisition(sealed)
    stage1_recipe = stage1.default_stage1_recipe(sealed, acquisition)
    final = synthetic_stage1_final(stage1_recipe)

    with tempfile.TemporaryDirectory(prefix="rift-se-b7873200-") as temporary:
        root = Path(temporary)
        bundle_path = root / stage1.STAGE1_FINAL_BUNDLE_FILENAME
        bundle = stage1.build_stage1_final_bundle(final, stage1_recipe)
        stage1.atomic_save_stage1_bundle(bundle, bundle_path)
        audited = stage1.validate_b7873200_stage1_final(bundle_path, expected_recipe=stage1_recipe)
        cloud = stage1.load_b7873200_stage1_cloud(bundle_path, expected_recipe=stage1_recipe)
        assert audited["audit"]["retained_count"] == 4 and cloud["points"].shape == (4, 3)
        reject("generic final masquerading as bundle", lambda: stage1.validate_b7873200_stage1_final(root / "checkpoint_final.pth.tar"))

        recipe = tiny_recipe()
        # Internal helpers accept this reduced engineering recipe, while the
        # public entrypoint still rejects every recipe other than frozen v1.
        geometry = trainer._stage1_geometry(cloud, recipe, torch.device("cpu"))
        model = analytic_fixture_model(recipe, geometry)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(42)
        checkpoint_contract = {
            "contract": {
                "method": trainer.METHOD_NAME,
                "campaign_identity": trainer.CAMPAIGN_IDENTITY,
                "artifact_identity": trainer.ARTIFACT_IDENTITY,
                "policy_identity": trainer.POLICY_IDENTITY,
                "implementation_kind": trainer.IMPLEMENTATION_KIND,
            },
            "synthetic": "checkpoint-contract-v1",
        }
        lifecycle_contract = {"synthetic": "lifecycle-contract-v1"}
        init_optimizer = torch.optim.Adam(model.parameters(), lr=float(recipe["init_lr"]))

        initial_state = trainer._checkpoint_state(
            contract_record=checkpoint_contract,
            recipe=recipe,
            phase=LifecyclePhase.INITIALIZATION,
            step=0,
            init_step=0,
            model=model,
            init_optimizer=init_optimizer,
            optimizer=None,
            scheduler=None,
            geometry=geometry,
            initialization=None,
            best_loss=float("inf"),
            history=[],
            gate_history=[],
            refresh_history=[],
            iso_points=None,
            iso_normals=None,
            generator=generator,
            resume_allowed=True,
        )
        assert trainer._validate_checkpoint(
            initial_state,
            contract_record=checkpoint_contract,
            recipe=recipe,
            geometry=geometry,
            model=model,
            require_resumable=True,
        ) is LifecyclePhase.INITIALIZATION
        changed = copy.deepcopy(initial_state)
        changed["resume_allowed"] = False
        reject(
            "nonresumable initialization checkpoint",
            lambda: trainer._validate_checkpoint(changed, contract_record=checkpoint_contract, recipe=recipe, geometry=geometry, model=model, require_resumable=True),
        )
        changed = copy.deepcopy(initial_state)
        changed["model_state_dict"]["output.weight"][0, 0] = float("nan")
        reject(
            "nonfinite checkpoint model",
            lambda: trainer._validate_checkpoint(changed, contract_record=checkpoint_contract, recipe=recipe, geometry=geometry, model=model, require_resumable=True),
        )

        def persist_unexpected(*_args) -> None:
            raise AssertionError("the data-free fixture did not request an interruption")

        init_step, init_optimizer, initialization = trainer._fit_initialization(
            model=model,
            geometry=geometry,
            recipe=recipe,
            generator=generator,
            device=torch.device("cpu"),
            start_step=0,
            optimizer=init_optimizer,
            persist_clean=persist_unexpected,
        )
        assert init_step == int(recipe["init_steps"]) and initialization["initial_gate"]["passed"] is True

        # Regression for SIGTERM immediately after the final initialization
        # update: the saved finite loss is gated without replaying an update.
        interrupted_initialization = copy.deepcopy(initialization)
        interrupted_initialization["initial_gate"] = None
        interrupted_state = trainer._checkpoint_state(
            contract_record=checkpoint_contract,
            recipe=recipe,
            phase=LifecyclePhase.INITIALIZATION,
            step=0,
            init_step=init_step,
            model=model,
            init_optimizer=init_optimizer,
            optimizer=None,
            scheduler=None,
            geometry=geometry,
            initialization=interrupted_initialization,
            best_loss=float("inf"),
            history=[],
            gate_history=[],
            refresh_history=[],
            iso_points=None,
            iso_normals=None,
            generator=generator,
            resume_allowed=True,
        )
        assert trainer._validate_checkpoint(
            interrupted_state,
            contract_record=checkpoint_contract,
            recipe=recipe,
            geometry=geometry,
            model=model,
            require_resumable=True,
        ) is LifecyclePhase.INITIALIZATION
        resumed_initialization = trainer._finalize_initialization(
            model=model,
            geometry=geometry,
            recipe=recipe,
            device=torch.device("cpu"),
            completed_steps=init_step,
            final_loss=interrupted_initialization["final_loss"],
        )
        assert resumed_initialization["initial_gate"]["passed"] is True

        optimizer = torch.optim.Adam(model.parameters(), lr=float(recipe["lr"]))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(recipe["steps"]), eta_min=float(recipe["lr"]) * 0.01
        )
        model.train()
        on = geometry["points"].clone()
        field, gradient = trainer.spatial_gradient(model, on, create_graph=True)
        one_step_loss = field.abs().mean() + 0.01 * (gradient.norm(dim=-1) - 1.0).square().mean()
        one_step_loss = one_step_loss + 1.0e-8 * sum(parameter.square().sum() for parameter in model.parameters())
        optimizer.zero_grad(set_to_none=True)
        one_step_loss.backward()
        optimizer.step()
        scheduler.step()
        trainer._assert_finite_model_and_optimizer(model, optimizer)

        model.eval()
        pre_refresh = trainer.strict_field_gate(
            model, geometry["extent"], int(recipe["gate_grid"]), float(recipe["boundary_margin"]),
            geometry["shell"], torch.device("cpu"), int(recipe["grid_chunk"]),
        )
        pre_refresh.update({"step": 1, "phase": "pre_iso_refresh"})
        assert pre_refresh["passed"] is True
        iso_points, refresh = trainer.refresh_iso_points_strict(
            model,
            geometry["points"],
            geometry["extent"],
            geometry["pitch"],
            int(recipe["n_iso"]),
            generator,
            oversample=float(recipe["projection_oversample"]),
        )
        iso_normals_np = trainer.estimate_pca_normals(
            iso_points.detach().cpu().numpy(), radius=3.0 * float(geometry["pitch"])
        )
        iso_normals_np, orientation = trainer.orient_normals_outward(
            iso_points.detach().cpu().numpy(), iso_normals_np, geometry["spec"].center
        )
        iso_normals = torch.as_tensor(iso_normals_np, dtype=torch.float32)
        post_refresh = trainer.strict_field_gate(
            model, geometry["extent"], int(recipe["gate_grid"]), float(recipe["boundary_margin"]),
            geometry["shell"], torch.device("cpu"), int(recipe["grid_chunk"]),
        )
        post_refresh.update({"step": 1, "phase": "post_iso_refresh"})
        refresh.update({"step": 1, "pre_gate": pre_refresh, "post_gate": post_refresh, "iso_normal_orientation": orientation})
        assert post_refresh["passed"] is True
        post_optimizer = trainer.strict_field_gate(
            model, geometry["extent"], int(recipe["gate_grid"]), float(recipe["boundary_margin"]),
            geometry["shell"], torch.device("cpu"), int(recipe["grid_chunk"]),
        )
        post_optimizer.update({"step": 1, "phase": "post_optimizer"})
        pre_export = trainer.strict_field_gate(
            model, geometry["extent"], int(recipe["gate_grid"]), float(recipe["boundary_margin"]),
            geometry["shell"], torch.device("cpu"), int(recipe["grid_chunk"]),
        )
        pre_export.update({"step": 1, "phase": "pre_export"})
        assert post_optimizer["passed"] is True and pre_export["passed"] is True

        gates = [
            {**initialization["initial_gate"], "step": 0, "phase": "post_initialization"},
            pre_refresh,
            post_refresh,
            post_optimizer,
            pre_export,
        ]
        history = [history_row(1, post_optimizer, optimizer, float(one_step_loss.detach()), len(iso_points))]
        pending = trainer._checkpoint_state(
            contract_record=checkpoint_contract,
            recipe=recipe,
            phase=LifecyclePhase.EXPORT_PENDING,
            step=1,
            init_step=init_step,
            model=model,
            init_optimizer=None,
            optimizer=optimizer,
            scheduler=scheduler,
            geometry=geometry,
            initialization=initialization,
            best_loss=float(one_step_loss.detach()),
            history=history,
            gate_history=gates,
            refresh_history=[refresh],
            iso_points=iso_points,
            iso_normals=iso_normals,
            generator=generator,
            resume_allowed=False,
        )
        assert trainer._validate_checkpoint(
            pending,
            contract_record=checkpoint_contract,
            recipe=recipe,
            geometry=geometry,
            model=model,
            require_resumable=False,
        ) is LifecyclePhase.EXPORT_PENDING

        run_root = root / "lifecycle"
        run_root.mkdir()
        latest = run_root / "checkpoint_latest.pth.tar"
        atomic_torch_save(interrupted_state, latest)
        clean = clean_interruption_record(
            phase=LifecyclePhase.INITIALIZATION,
            step=0,
            max_steps=int(recipe["steps"]),
            checkpoint_path=str(latest),
            expected_contract=lifecycle_contract,
        )
        write_lifecycle(run_root / "lifecycle.json", clean, expected_contract=lifecycle_contract, max_steps=int(recipe["steps"]))
        assert prepare_run_root(run_root, resume=latest, expected_contract=lifecycle_contract, max_steps=int(recipe["steps"])) == "resume"
        bad_lifecycle = dict(clean)
        bad_lifecycle["resume_allowed"] = "yes"
        reject(
            "nonboolean lifecycle resume flag",
            lambda: validate_lifecycle_record(bad_lifecycle, expected_contract=lifecycle_contract, max_steps=int(recipe["steps"])),
        )
        assert same_value(torch.tensor([1.0]), torch.tensor([1.0]))
        assert not same_value(torch.tensor([1.0]), torch.tensor([2.0]))

        staging = root / "staging"
        staging.mkdir()
        surface_audit = trainer._export_surface(
            model=model, geometry=geometry, recipe=recipe, device=torch.device("cpu"), path=staging / "surface_reconstruction.npz"
        )
        final_state = trainer._completion_checkpoint(pending, surface_audit)
        atomic_torch_save(final_state, staging / "checkpoint_final.pth.tar")
        atomic_json_dump(
            {
                "method": trainer.METHOD_NAME,
                "campaign_identity": trainer.CAMPAIGN_IDENTITY,
                "artifact_identity": trainer.ARTIFACT_IDENTITY,
                "policy_identity": trainer.POLICY_IDENTITY,
                "implementation_kind": trainer.IMPLEMENTATION_KIND,
                "contract": lifecycle_contract,
                "steps": 1,
                "best_loss": float(one_step_loss.detach()),
                "final_gate": pre_export,
                "last_iso_refresh": refresh,
                "surface_audit": surface_audit,
                "ground_truth_geometry_used": False,
                "novel_view_signal_supported": False,
            },
            staging / "run_summary.json",
        )
        atomic_json_dump(
            {
                "schema": trainer.CHECKPOINT_SCHEMA,
                "phase": "complete",
                "state": "complete",
                "step": 1,
                "max_steps": 1,
                "resume_allowed": False,
                "contract": lifecycle_contract,
            },
            staging / "status.json",
        )

        # The ordinary reconstruction is deliberately rejected because its
        # random MLP cannot match the analytic fixture field.  That proves the
        # completion validator binds the saved field to the checkpoint model.
        reject(
            "surface unrelated to default model",
            lambda: trainer._validate_complete_package(
                staging,
                checkpoint_contract=checkpoint_contract,
                lifecycle_contract=lifecycle_contract,
                recipe=recipe,
                geometry=geometry,
                device=torch.device("cpu"),
            ),
        )
        original_model = trainer._model
        trainer._model = lambda _recipe, _extent, _device: model
        try:
            trainer._validate_complete_package(
                staging,
                checkpoint_contract=checkpoint_contract,
                lifecycle_contract=lifecycle_contract,
                recipe=recipe,
                geometry=geometry,
                device=torch.device("cpu"),
            )
            complete = promote_complete_package(
                staging,
                root / "complete",
                validate=lambda path: trainer._validate_complete_package(
                    path,
                    checkpoint_contract=checkpoint_contract,
                    lifecycle_contract=lifecycle_contract,
                    recipe=recipe,
                    geometry=geometry,
                    device=torch.device("cpu"),
                ),
            )
            assert complete.is_dir() and not staging.exists()
        finally:
            trainer._model = original_model
        actual_entrypoint_recovery_checks(root=root / "entrypoint", source=cloud, recipe=recipe)
    print("SE_B7873200_STAGE2_CPU_CONTRACT_PASS: 36 allocated synthetic checks", flush=True)


if __name__ == "__main__":
    main()
