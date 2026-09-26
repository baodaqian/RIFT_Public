from __future__ import annotations

from dataclasses import FrozenInstanceError
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest import mock

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "rift"
    / "sugavanam_ertin_stage2_stabilized_v3.py"
)
SPEC = importlib.util.spec_from_file_location("se_stage2_stabilized_v3_contract", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
v3 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = v3
SPEC.loader.exec_module(v3)


def stage1_state(scale: float = 1.0) -> dict[str, object]:
    shape = v3.STAGE1_GRID_SHAPE
    real = np.ones(shape, dtype=np.float32) * scale
    imag = np.zeros(shape, dtype=np.float32)
    active = np.ones(shape, dtype=bool)
    positions = np.zeros((*shape, 3), dtype=np.float32)
    return {
        "scene_repr": "grid",
        "epoch": 150,
        "granularity": 48,
        "extent": 0.15,
        "model_state_dict": {
            "w_re": real,
            "w_im": imag,
            "active_mask": active,
            "grid_positions": positions,
        },
    }


def checkpoint_record(contract, phase=v3.LifecyclePhase.TRAINING, resumable=True):
    record = {
        **v3.identity_record(contract),
        "phase": phase.value,
        "step": 4,
        "max_steps": 10,
        "resume_allowed": resumable,
        "checkpoint_role": "latest" if phase is not v3.LifecyclePhase.COMPLETE else "final",
    }
    if resumable:
        record.update(
            {
                "model_state_dict": {"weights": np.asarray([1.0])},
                "optimizer_state_dict": {"param_groups": [{}]},
                "rng_state": {"python": "state"},
                "sample_rng_state": np.asarray([1], dtype=np.uint8),
                "stage1_audit": {
                    "scene_key": contract.key,
                    "checkpoint_path": contract.stage1_checkpoint,
                    "epoch": 150,
                    "granularity": 48,
                    "extent": 0.15,
                },
                "model_config": {"n_fourier": 9},
                "args": {
                    "scene_key": contract.key,
                    "measurement_path": contract.measurement_path,
                    "stage1_checkpoint": contract.stage1_checkpoint,
                    "output_dir": contract.output_dir,
                },
            }
        )
        if phase is v3.LifecyclePhase.INITIALIZATION:
            record["initialization_state"] = {"step": 4}
        elif phase is v3.LifecyclePhase.TRAINING:
            record.update(
                {
                    "scheduler_state_dict": {"last_epoch": 4},
                    "initialization": {"passed": True},
                    "history": [],
                }
            )
        elif phase is v3.LifecyclePhase.EXPORT_PENDING:
            record.update(
                {
                    "initialization": {"passed": True},
                    "history": [],
                    "final_gate": {"passed": True},
                }
            )
    return record


class StabilizedStage2V3ContractTests(unittest.TestCase):
    def test_registry_is_exact_frozen_and_b787_is_sphere2k(self) -> None:
        self.assertEqual(
            tuple(v3.SCENE_CONTRACTS),
            ("a320", "x59", "firetruck", "racecar", "loader", "b787_sphere2k"),
        )
        self.assertEqual(
            {contract.acquisition for contract in list(v3.SCENE_CONTRACTS.values())[:5]},
            {"sphere10k"},
        )
        b787 = v3.get_scene_contract("b787_sphere2k")
        self.assertEqual(b787.acquisition, "sphere2k")
        self.assertTrue(b787.measurement_path.endswith("r10m_sphere2k.npz"))
        self.assertTrue(
            b787.stage1_checkpoint.endswith(
                "/training_checkpoints/b787_sugavanam_ertin_scatter/checkpoint_final.pth.tar"
            )
        )
        self.assertEqual(
            b787.output_dir,
            "/storage/scratch1/1/dbao31/"
            "rift_sugavanam_ertin_stage2_stabilized_v3/b787_sphere2k",
        )
        with self.assertRaises(TypeError):
            v3.SCENE_CONTRACTS["extra"] = b787
        with self.assertRaises(FrozenInstanceError):
            b787.label = "changed"

    def test_public_scenes_and_path_overrides_fail_closed(self) -> None:
        for scene in ("camry", "gotcha", "honda", "jeep", "rpd_00", "unknown"):
            with self.subTest(scene=scene), self.assertRaises(v3.ContractViolation):
                v3.get_scene_contract(scene)
        contract = v3.get_scene_contract("a320")
        v3.validate_claimed_paths(
            contract,
            measurement_path=contract.measurement_path,
            stage1_checkpoint=contract.stage1_checkpoint,
            output_dir=contract.output_dir,
        )
        with self.assertRaises(v3.ContractViolation):
            v3.validate_claimed_paths(
                contract,
                measurement_path=contract.measurement_path,
                stage1_checkpoint=contract.stage1_checkpoint.replace("checkpoint_final", "checkpoint_latest"),
                output_dir=contract.output_dir,
            )

    def test_only_exact_registry_contract_objects_are_accepted(self) -> None:
        canonical = v3.get_scene_contract("a320")
        clone = v3.SceneContract(**canonical.as_dict())
        public = v3.SceneContract(
            key="camry",
            label="Camry",
            acquisition="public",
            measurement_path="/tmp/public.npz",
            stage1_checkpoint="/tmp/checkpoint_final.pth.tar",
            output_dir="/tmp/public-output",
        )
        for forged in (clone, public):
            with self.subTest(scene=forged.key), self.assertRaises(v3.ContractViolation):
                v3.identity_record(forged)
            with self.subTest(scene=forged.key), self.assertRaises(v3.ContractViolation):
                v3.validate_claimed_paths(
                    forged,
                    measurement_path=forged.measurement_path,
                    stage1_checkpoint=forged.stage1_checkpoint,
                    output_dir=forged.output_dir,
                )
        self.assertFalse(hasattr(v3, "_SCENE_CONTRACTS"))

    def test_stage1_final_structure_allows_scene_derived_thresholds(self) -> None:
        a320 = v3.get_scene_contract("a320")
        b787 = v3.get_scene_contract("b787_sphere2k")
        first = v3.validate_stage1_final_state(
            stage1_state(1.0), a320, source_path=a320.stage1_checkpoint
        )
        second = v3.validate_stage1_final_state(
            stage1_state(2.0), b787, source_path=b787.stage1_checkpoint
        )
        self.assertEqual(first.retained_count, 48**3)
        self.assertEqual(second.retained_count, 48**3)
        self.assertAlmostEqual(first.threshold_absolute, 0.15)
        self.assertAlmostEqual(second.threshold_absolute, 0.30)
        self.assertNotEqual(first.threshold_absolute, second.threshold_absolute)

    def test_stage1_loader_can_only_receive_registry_final(self) -> None:
        contract = v3.get_scene_contract("loader")
        seen = []

        def load(path: str):
            seen.append(path)
            return stage1_state()

        audit = v3.validate_stage1_final_checkpoint(
            contract, load, is_file=lambda path: path == contract.stage1_checkpoint
        )
        self.assertEqual(seen, [contract.stage1_checkpoint])
        self.assertEqual(audit.scene_key, "loader")
        with self.assertRaises(v3.ContractViolation):
            v3.validate_stage1_final_checkpoint(contract, load, is_file=lambda _path: False)

    def test_stage1_rejects_latest_wrong_epoch_shape_and_nonfinite_state(self) -> None:
        contract = v3.get_scene_contract("a320")
        with self.assertRaises(v3.ContractViolation):
            v3.validate_stage1_final_state(
                stage1_state(),
                contract,
                source_path=contract.stage1_checkpoint.replace("checkpoint_final", "checkpoint_latest"),
            )
        wrong_epoch = stage1_state()
        wrong_epoch["epoch"] = 149
        with self.assertRaises(v3.ContractViolation):
            v3.validate_stage1_final_state(
                wrong_epoch, contract, source_path=contract.stage1_checkpoint
            )
        wrong_shape = stage1_state()
        wrong_shape["model_state_dict"]["w_re"] = np.ones((4, 4, 4), dtype=np.float32)
        with self.assertRaises(v3.ContractViolation):
            v3.validate_stage1_final_state(
                wrong_shape, contract, source_path=contract.stage1_checkpoint
            )
        nonfinite = stage1_state()
        nonfinite["model_state_dict"]["w_im"][0, 0, 0] = np.nan
        with self.assertRaises(v3.ContractViolation):
            v3.validate_stage1_final_state(
                nonfinite, contract, source_path=contract.stage1_checkpoint
            )

    def test_checkpoint_identity_rejects_cross_scene_resume(self) -> None:
        a320 = v3.get_scene_contract("a320")
        x59 = v3.get_scene_contract("x59")
        state = checkpoint_record(a320)
        self.assertEqual(
            v3.validate_checkpoint_identity(state, a320, require_resumable=True),
            v3.LifecyclePhase.TRAINING,
        )
        with self.assertRaises(v3.ContractViolation):
            v3.validate_checkpoint_identity(state, x59, require_resumable=True)
        state["resume_allowed"] = False
        with self.assertRaises(v3.ContractViolation):
            v3.validate_checkpoint_identity(state, a320, require_resumable=True)

    def test_resumable_checkpoint_requires_latest_role_and_phase_payload(self) -> None:
        contract = v3.get_scene_contract("loader")
        for phase in (
            v3.LifecyclePhase.INITIALIZATION,
            v3.LifecyclePhase.TRAINING,
            v3.LifecyclePhase.EXPORT_PENDING,
        ):
            with self.subTest(phase=phase):
                valid = checkpoint_record(contract, phase=phase, resumable=True)
                self.assertEqual(
                    v3.validate_checkpoint_identity(valid, contract, require_resumable=True),
                    phase,
                )
                missing_role = dict(valid)
                missing_role.pop("checkpoint_role")
                with self.assertRaises(v3.ContractViolation):
                    v3.validate_checkpoint_identity(
                        missing_role, contract, require_resumable=True
                    )
                missing_payload = dict(valid)
                missing_payload.pop("model_state_dict")
                with self.assertRaises(v3.ContractViolation):
                    v3.validate_checkpoint_identity(
                        missing_payload, contract, require_resumable=True
                    )

    def test_clean_term_status_is_resumable_and_exits_143(self) -> None:
        contract = v3.get_scene_contract("firetruck")
        for phase in (
            v3.LifecyclePhase.INITIALIZATION,
            v3.LifecyclePhase.TRAINING,
            v3.LifecyclePhase.EXPORT_PENDING,
        ):
            with self.subTest(phase=phase):
                status = v3.clean_term_status(contract, phase, step=3, max_steps=10)
                self.assertEqual(v3.validate_status_record(status, contract), phase)
                with self.assertRaises(SystemExit) as stopped:
                    v3.exit_after_clean_term(status, contract)
                self.assertEqual(stopped.exception.code, 143)
        with self.assertRaises(v3.ContractViolation):
            v3.clean_term_status(
                contract, v3.LifecyclePhase.COMPLETE, step=10, max_steps=10
            )

    def test_invalid_lifecycle_states_fail_closed(self) -> None:
        contract = v3.get_scene_contract("racecar")
        complete_running = {
            **v3.identity_record(contract),
            "phase": "complete",
            "state": "running",
            "step": 10,
            "max_steps": 10,
            "resume_allowed": False,
        }
        with self.assertRaises(v3.ContractViolation):
            v3.validate_status_record(complete_running, contract)
        bad_progress = checkpoint_record(contract)
        bad_progress["step"] = 11
        with self.assertRaises(v3.ContractViolation):
            v3.validate_checkpoint_identity(bad_progress, contract)
        fractional_progress = checkpoint_record(contract)
        fractional_progress["step"] = 4.5
        with self.assertRaises(v3.ContractViolation):
            v3.validate_checkpoint_identity(fractional_progress, contract)
        failed_complete = {
            **v3.identity_record(contract),
            "phase": "complete",
            "state": "failed",
            "step": 10,
            "max_steps": 10,
            "resume_allowed": False,
            "exit_code": 1,
        }
        with self.assertRaises(v3.ContractViolation):
            v3.validate_status_record(failed_complete, contract)
        failed_zero = {
            **failed_complete,
            "phase": "training",
            "exit_code": 0,
        }
        with self.assertRaises(v3.ContractViolation):
            v3.validate_status_record(failed_zero, contract)
        running_resumable = {
            **failed_complete,
            "phase": "training",
            "state": "running",
            "resume_allowed": True,
            "exit_code": None,
        }
        with self.assertRaises(v3.ContractViolation):
            v3.validate_status_record(running_resumable, contract)

    def test_complete_artifact_set_is_verified_idempotent_noop(self) -> None:
        contract = v3.get_scene_contract("b787_sphere2k")
        final = checkpoint_record(contract, v3.LifecyclePhase.COMPLETE, resumable=False)
        final["step"] = final["max_steps"]
        final["surface_audit"] = {
            "validity": {"passed": True},
            "topology": {"passed": True},
        }
        surface = {
            **v3.identity_record(contract),
            "phase": "complete",
            "step": 10,
            "max_steps": 10,
            "validity_passed": np.asarray(True),
            "topology_passed": np.bool_(True),
        }
        summary = {
            **v3.identity_record(contract),
            "phase": "complete",
            "step": 10,
            "max_steps": 10,
            "checkpoint_final": contract.final_checkpoint,
            "surface_reconstruction": contract.surface_path,
        }
        status = {
            **v3.identity_record(contract),
            "phase": "complete",
            "state": "complete",
            "step": 10,
            "max_steps": 10,
            "resume_allowed": False,
            "exit_code": 0,
            "checkpoint_path": contract.final_checkpoint,
        }
        disposition = v3.validate_idempotent_complete_noop(
            v3.COMPLETE_ARTIFACTS,
            final_checkpoint=final,
            surface_metadata=surface,
            summary=summary,
            status=status,
            contract=contract,
        )
        self.assertEqual(disposition, v3.OutputDisposition.COMPLETE_NOOP)
        partial_final = dict(final)
        partial_final["step"] = 9
        with self.assertRaises(v3.ContractViolation):
            v3.validate_idempotent_complete_noop(
                v3.COMPLETE_ARTIFACTS,
                final_checkpoint=partial_final,
                surface_metadata=surface,
                summary=summary,
                status=status,
                contract=contract,
            )
        mismatched_summary = dict(summary)
        mismatched_summary["max_steps"] = 11
        with self.assertRaises(v3.ContractViolation):
            v3.validate_idempotent_complete_noop(
                v3.COMPLETE_ARTIFACTS,
                final_checkpoint=final,
                surface_metadata=surface,
                summary=mismatched_summary,
                status=status,
                contract=contract,
            )
        self.assertEqual(
            v3.complete_artifact_disposition({"status.json", "checkpoint_latest.pth.tar"}),
            v3.OutputDisposition.NOT_COMPLETE,
        )
        with self.assertRaises(v3.ContractViolation):
            v3.complete_artifact_disposition(
                v3.COMPLETE_ARTIFACTS - {"run_summary.json"}
            )
        with self.assertRaises(v3.ContractViolation):
            v3.complete_artifact_disposition(
                v3.COMPLETE_ARTIFACTS | {"terminal_failure.json"}
            )

    def test_geometry_exports_delegate_to_preserved_module(self) -> None:
        sentinel = object()
        preserved = SimpleNamespace(orient_normals_outward=sentinel)
        self.assertIn("orient_normals_outward", v3.GEOMETRY_EXPORT_NAMES)
        with mock.patch.object(v3, "import_module", return_value=preserved):
            self.assertIs(v3.__getattr__("orient_normals_outward"), sentinel)
        with self.assertRaises(AttributeError):
            v3.__getattr__("not_a_geometry_export")


if __name__ == "__main__":
    unittest.main()
