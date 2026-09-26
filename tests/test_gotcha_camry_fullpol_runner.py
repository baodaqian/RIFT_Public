"""Torch-free integration checks for the Camry full-pol runnable layer."""

from __future__ import annotations

import ast
import importlib.util
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load(ROOT / "scripts" / "run_gotcha_camry_fullpol_v1.py", "gotcha_camry_fullpol_runner_tests")
VALIDATOR = _load(ROOT / "scripts" / "validate_gotcha_camry_fullpol_v1.py", "gotcha_camry_fullpol_validator_tests")
RUNTIME_VALIDATOR = _load(ROOT / "scripts" / "validate_gotcha_camry_fullpol_runtime.py", "gotcha_camry_fullpol_runtime_validator_tests")


class CamryFullPolRunnerTest(unittest.TestCase):
    def _fake_shard(self, polarization: str, *, include_test: bool = False):
        identity_type = RUNNER.ACQ.NativeObservationId
        identities = (
            identity_type(1, polarization, 2, 0),
            identity_type(1, polarization, 2, 1),
            identity_type(1, polarization, 3, 2),
        )
        roles = np.asarray(["train", "train", "test" if include_test else "validation"], dtype="U10")
        co_pol = polarization in {"hh", "vv"}
        autofocus = SimpleNamespace(
            schema="rift_gotcha_autofocus_provenance_v1",
            mode="raw_channel_own_arrays_unapplied" if co_pol else "official_arrays_absent",
            official_available=co_pol,
            applied=False,
            source_shard_id=f"pass1_{polarization}" if co_pol else None,
            range_field="r_correct_raw" if co_pol else None,
            phase_field="ph_correct_raw" if co_pol else None,
        )
        phase_reference = SimpleNamespace(
            schema="rift_gotcha_native_phase_reference_v1",
            frequency_unit="Hz",
            position_unit="m",
            range_unit="m",
            phase_unit="rad",
            frequency_values="native_stored_exact",
            reference_range_field="r0",
            geometry_contract="paired_monostatic_tx_equals_rx_same_observation",
            correction_application_convention="unverified_no_default",
        )
        corrections = np.asarray([0.1, 0.2, 0.3], dtype=np.float64) if co_pol else np.empty(0, dtype=np.float64)
        return SimpleNamespace(
            shard_id=f"pass1_{polarization}",
            polarization=polarization,
            pass_id=1,
            observation_ids=identities,
            role=roles,
            frequencies_hz=np.asarray([9.0e9, 10.0e9, 11.0e9], dtype=np.float64),
            view_count=3,
            autofocus=autofocus,
            phase_reference=phase_reference,
            r_correct_raw=corrections,
            ph_correct_raw=corrections,
        )

    def test_protocol_and_frozen_config(self):
        protocol = RUNNER.load_protocol(ROOT / "protocols" / "gotcha_camry_fullpol_v1.json")
        self.assertEqual(protocol["schema"], RUNNER.PROTOCOL_SCHEMA)
        self.assertEqual(protocol["selection"]["polarizations"], list(RUNNER.POLARIZATIONS))
        self.assertTrue(protocol["test_sealed"])
        self.assertEqual(protocol["outputs"]["config"], "config.json")
        self.assertEqual(protocol["outputs"]["device"], "device.json")
        self.assertEqual(protocol["outputs"]["cgls_metrics"], "cgls_metrics.json")
        self.assertIn("checkpoint_initialization.pt", protocol["checkpoints"]["required"])
        RUNNER.FROZEN_CONFIG.validate()
        self.assertEqual(RUNNER.FROZEN_CONFIG.epochs, 150)
        self.assertEqual(RUNNER.FROZEN_CONFIG.max_active_sites, 8192)
        self.assertEqual(RUNNER.FROZEN_CONFIG.as_dict()["scheduler"]["T0"], 10)
        self.assertEqual(protocol["resource"]["qos"], "inferno")
        self.assertEqual(protocol["resource"]["partition"], "gpu-a100")
        self.assertEqual(protocol["resource"]["gpu"], "A100")
        self.assertEqual(protocol["resource"]["cpus"], 8)
        self.assertEqual(protocol["resource"]["walltime"], "2-00:00:00")
        self.assertTrue(protocol["validation"]["same_allocation_before_fit"])
        self.assertTrue(protocol["validation"]["cuda_skip_is_failure"])
        self.assertEqual(protocol["validation"]["runtime_validator"], "scripts/validate_gotcha_camry_fullpol_runtime.py")
        self.assertEqual(protocol["validation"]["runtime_command"], "python3 -B scripts/validate_gotcha_camry_fullpol_runtime.py")
        self.assertEqual(protocol["validation"]["expected_core_test_count"], 19)
        self.assertIn("test_cuda_checkpoint_round_trip_restores_rng_optimizer_devices_and_resume_readiness", protocol["validation"]["cuda_required_test"])

    def test_static_validator_is_torch_free_and_implicit(self):
        result = VALIDATOR.validate_static(ROOT / "protocols" / "gotcha_camry_fullpol_v1.json")
        self.assertTrue(result["status"].startswith("PASS_STATIC"))
        self.assertFalse(result["torch_imported"])
        self.assertFalse(result["dense_N3_materialized"])
        self.assertFalse(result["archive_accessed"])
        self.assertTrue(result["static_checks"]["core_contract"])

    def test_runtime_validator_preserves_failed_test_tracebacks(self):
        class FakeTest:
            def id(self):
                return "tests.fake_case"

        self.assertEqual(
            RUNTIME_VALIDATOR._test_details(((FakeTest(), "Traceback (most recent call last):\\nValueError"),)),
            [{"id": "tests.fake_case", "traceback": "Traceback (most recent call last):\\nValueError"}],
        )

    def test_block_representatives_are_legal_without_dense_lattice(self):
        indices, blocks = RUNNER._block_representatives(801, 16)
        self.assertEqual(indices.shape[0], len(blocks))
        self.assertLess(indices.shape[0], 801**3)
        self.assertTrue(np.all(indices >= 0))
        self.assertTrue(np.all(indices < 801))
        self.assertEqual(len({tuple(row) for row in indices.tolist()}), indices.shape[0])

    def test_refinement_is_bounded_and_on_lattice(self):
        indices = RUNNER._refined_representatives(((0, 0, 0), (3, 5, 7)), 801, 16, max_bisections=1)
        self.assertLessEqual(indices.shape[0], 16)
        self.assertTrue(np.all(indices >= 0))
        self.assertTrue(np.all(indices < 801))
        self.assertEqual(len({tuple(row) for row in indices.tolist()}), indices.shape[0])

    def test_header_inventory_is_channel_owned_and_test_sealed(self):
        inventories = [RUNNER._selected_header_inventory(self._fake_shard(polarization)) for polarization in RUNNER.POLARIZATIONS]
        self.assertEqual({entry["polarization"] for entry in inventories}, set(RUNNER.POLARIZATIONS))
        self.assertTrue(all(entry["selected_record_count"] == 2 for entry in inventories))
        self.assertEqual(
            {entry["polarization"]: entry["source_af_representation"] for entry in inventories},
            {"hh": "own_source_af_once", "hv": "raw_unapplied", "vh": "raw_unapplied", "vv": "own_source_af_once"},
        )
        with self.assertRaises(ValueError):
            RUNNER._selected_header_inventory(self._fake_shard("vv", include_test=True))

    def test_complete_preflight_resource_report_uses_live_scout_bounds(self):
        class SyntheticPath:
            def __init__(self, name):
                self.name = name

            def is_symlink(self):
                return False

            def is_file(self):
                return True

            def __str__(self):
                return f"/synthetic/{self.name}"

        shards = {polarization: self._fake_shard(polarization) for polarization in RUNNER.POLARIZATIONS}

        def fake_loader(path, **kwargs):
            self.assertEqual(kwargs, {"expected_pass_id": 1, "expected_scene_id": "gotcha_v1_joint8_fullpol"})
            return shards[path.name.removeprefix("pass1_").removesuffix(".npz")]

        paths = tuple(SyntheticPath(f"pass1_{polarization}.npz") for polarization in RUNNER.POLARIZATIONS)
        with mock.patch.object(RUNNER.ACQ, "load_native_shard", side_effect=fake_loader):
            report, loaded = RUNNER.preflight_archives(paths)
        self.assertEqual(len(loaded), 4)
        resource = report["resource"]
        fmax = 11.0e9
        h_space = RUNNER.SPEED_OF_LIGHT_M_S / (2.0 * fmax)
        N = int(np.floor(10.0 / h_space)) + 1
        block_count = int(np.ceil(N / RUNNER.SCOUT_BLOCK_WIDTH)) ** 3
        top_count = min(RUNNER.SCOUT_INTERNAL_BLOCK_QUOTA, block_count)
        refinement_parent_bound = max(RUNNER.SCOUT_MAX_FINAL_LEAVES - top_count, 0)
        terminal_leaf_bound = min(RUNNER.SCOUT_MAX_FINAL_LEAVES, top_count + 7 * refinement_parent_bound)
        scoring_bound = block_count + 8 * refinement_parent_bound
        total_samples = 4 * 2 * 3
        self.assertEqual(resource["qos"], "inferno")
        self.assertEqual(resource["partition"], "gpu-a100")
        self.assertEqual(resource["gpu"], "A100")
        self.assertEqual(resource["resource_ceiling"]["cpus"], 8)
        self.assertEqual(resource["resource_ceiling"]["walltime"], "2-00:00:00")
        self.assertEqual(resource["logical_batches_per_epoch_U"], 2)
        self.assertEqual(resource["scout"]["terminal_leaf_bound"], terminal_leaf_bound)
        self.assertEqual(resource["scout"]["scored_candidate_evaluation_bound"], scoring_bound)
        self.assertEqual(resource["work_estimate_direct_terms"]["scout_dc_adjoint"], scoring_bound * total_samples)
        self.assertEqual(resource["memory_estimate_bytes"]["scout_terminal_leaf_indices_upper_bound"], terminal_leaf_bound * 3 * 8)
        self.assertEqual(resource["memory_estimate_bytes"]["scout_scored_candidate_indices_upper_bound"], scoring_bound * 3 * 8)
        work = resource["work_estimate_direct_terms"]
        self.assertNotIn("active_operator_cgls", work)
        self.assertEqual(work["cgls_three_pass_recurrence"], 3 * RUNNER.MAX_ACTIVE_SITES * total_samples * 24)
        self.assertEqual(
            work["subtotal_before_scout_and_io"],
            (2 * RUNNER.EPOCHS + RUNNER.EPOCHS) * RUNNER.MAX_ACTIVE_SITES * total_samples + 3 * RUNNER.MAX_ACTIVE_SITES * total_samples * 24,
        )

    def test_archive_resolution_requires_exactly_four_distinct_paths(self):
        paths = RUNNER.resolve_archive_paths(None, "C:/pace/converted_v3_joint8_fullpol")
        self.assertEqual(len(paths), 4)
        self.assertEqual([path.name for path in paths], [f"pass1_{p}.npz" for p in RUNNER.POLARIZATIONS])
        with self.assertRaises(ValueError):
            RUNNER.resolve_archive_paths(["a.npz", "b.npz"], None)

    def test_non_dry_cli_passes_preflight_shards_to_run_without_name_error(self):
        loaded_shards = (object(), object(), object(), object())
        preflight = {
            "resource": {"qos": "inferno", "partition": "gpu-a100", "gpu": "A100"},
            "channels": {},
            "spatial": {},
        }
        observed = {}

        def fake_run(protocol, paths, output, **kwargs):
            observed.update(kwargs)
            return {"status": "PASS", "output": str(output)}

        with mock.patch.object(RUNNER, "preflight_archives", return_value=(preflight, loaded_shards)), mock.patch.object(RUNNER, "run_experiment", side_effect=fake_run), mock.patch.object(RUNNER, "_write_json"):
            status = RUNNER.main(["--archive-root", str(ROOT / "archives"), "--output", str(ROOT)])
        self.assertEqual(status, 0)
        self.assertIs(observed["shards"], loaded_shards)
        self.assertIsNone(observed["device"])
        self.assertFalse(observed["test_device"])

    def test_device_contract_requires_cuda_a100_for_production_and_explicit_cpu_for_tests(self):
        class FakeDevice:
            def __init__(self, value):
                text = str(value)
                self.type = text.split(":", 1)[0]
                self.index = None if ":" not in text else int(text.split(":", 1)[1])

            def __str__(self):
                return "cuda" if self.index is None and self.type == "cuda" else (self.type if self.index is None else f"{self.type}:{self.index}")

        class FakeProperties:
            total_memory = 141 * 1024**3
            major = 9
            minor = 0
            multi_processor_count = 132

        class FakeCuda:
            def __init__(self, name="NVIDIA A100"):
                self.name = name

            def is_available(self):
                return True

            def current_device(self):
                return 0

            def get_device_name(self, index):
                return self.name

            def get_device_properties(self, index):
                return FakeProperties()

            def device_count(self):
                return 1

        class FakeTorch:
            def __init__(self, name="NVIDIA A100"):
                self.cuda = FakeCuda(name)

            @staticmethod
            def device(value):
                return FakeDevice(value)

        protocol = {"resource": {"gpu": "A100"}}
        device, report = RUNNER.select_execution_device(FakeTorch(), None, protocol=protocol)
        self.assertEqual(str(device), "cuda")
        self.assertEqual(report["actual_device"], "cuda:0")
        self.assertTrue(report["a100_compatible"])
        with self.assertRaises(RuntimeError):
            RUNNER.select_execution_device(FakeTorch(), "cpu", protocol=protocol)
        cpu, cpu_report = RUNNER.select_execution_device(FakeTorch(), "cpu", protocol=protocol, test_device=True)
        self.assertEqual(str(cpu), "cpu")
        self.assertEqual(cpu_report["mode"], "explicit_test_device")
        with self.assertRaises(RuntimeError):
            RUNNER.select_execution_device(FakeTorch("NVIDIA H100"), None, protocol=protocol)
        with self.assertRaises(RuntimeError):
            RUNNER.select_execution_device(FakeTorch("NVIDIA H200"), None, protocol=protocol)
        with self.assertRaises(RuntimeError):
            RUNNER.select_execution_device(FakeTorch("NVIDIA L40S"), None, protocol=protocol)

    def test_scout_scores_are_length_k_and_energy_normalized(self):
        class FakeProjector:
            def target(self):
                return SimpleNamespace(values=(np.asarray([1.0 + 0.0j]),))

            def apply_adjoint(self, target):
                return target

        class FakeLattice:
            def points_native(self, indices):
                return np.asarray(indices, dtype=np.float64)

        class FakeOperator:
            def __init__(self, points_native, point_chunk_size):
                self.count = len(points_native)

            def adjoint_dc(self, records, residuals):
                return np.arange(1, self.count + 1, dtype=np.float64).astype(np.complex128)

        fake_core = SimpleNamespace(CamryL3SparseOperator=FakeOperator)
        projectors = {polarization: FakeProjector() for polarization in RUNNER.POLARIZATIONS}
        panel = SimpleNamespace(records=lambda polarization: ())
        scores = RUNNER._scout_scores(
            fake_core,
            panel,
            FakeLattice(),
            projectors,
            np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.int64),
            chunk_size=2,
        )
        np.testing.assert_allclose(scores, np.asarray([4.0, 16.0]))

    def test_scout_uses_terminal_leaves_and_counts_adaptive_scoring_passes(self):
        calls = []

        def fake_scores(core, panel, lattice, projectors, indices):
            calls.append(np.asarray(indices).copy())
            return np.arange(len(indices), 0, -1, dtype=np.float64)

        class FakeLattice:
            N = 257

        def rank_candidate_sites(indices, scores, *, max_active):
            self.assertLessEqual(len(indices), max_active)
            return np.asarray(indices)

        fake_core = SimpleNamespace(rank_candidate_sites=rank_candidate_sites)
        with mock.patch.object(RUNNER, "_scout_scores", side_effect=fake_scores):
            selected, report = RUNNER.scout_shared_support(fake_core, None, FakeLattice(), {})
        self.assertEqual(calls[0].shape[0], 17**3)
        self.assertGreaterEqual(len(calls), 2)
        self.assertFalse(report["representatives_rescored"])
        self.assertEqual(report["top_blocks_scored"], 2048)
        self.assertEqual(report["scored_candidate_count"], sum(batch.shape[0] for batch in calls))
        self.assertEqual(report["scoring_passes"], len(calls))
        self.assertEqual(report["refinement_stopping_reason"], "terminal_leaf_cap_reached")
        self.assertEqual(report["terminal_leaf_count"], len(selected))
        self.assertLessEqual(report["terminal_leaf_count"], RUNNER.SCOUT_MAX_FINAL_LEAVES)
        self.assertLessEqual(report["refinement_bisections"], RUNNER.SCOUT_MAX_BISECTIONS)

    def test_scout_edge_depth_two_work_stays_below_safe_reported_bounds(self):
        calls = []

        def fake_scores(core, panel, lattice, projectors, indices):
            calls.append(np.asarray(indices).copy())
            return np.ones((len(indices),), dtype=np.float64)

        class FakeLattice:
            N = 257

        def rank_candidate_sites(indices, scores, *, max_active):
            return np.asarray(indices)

        fake_core = SimpleNamespace(rank_candidate_sites=rank_candidate_sites)
        with mock.patch.object(RUNNER, "_scout_scores", side_effect=fake_scores):
            selected, report = RUNNER.scout_shared_support(fake_core, None, FakeLattice(), {})
        self.assertGreaterEqual(report["refinement_bisections"], 2)
        self.assertGreater(report["refinement_parent_count"], 877)
        self.assertLessEqual(report["refinement_parent_count"], report["refinement_parent_bound"])
        self.assertLessEqual(report["terminal_leaf_count"], report["terminal_leaf_bound"])
        self.assertLessEqual(report["scored_candidate_count"], report["scored_candidate_evaluation_bound"])
        self.assertEqual(report["scored_candidate_count"], sum(batch.shape[0] for batch in calls))
        self.assertEqual(report["scoring_passes"], len(calls))
        self.assertEqual(report["terminal_leaf_count"], len(selected))

    def test_optimizer_improvement_reports_signed_channel_reductions(self):
        initial = {
            "range_macro_relmse": 0.5,
            "range_relmse_by_polarization": {polarization: 0.5 + index * 0.1 for index, polarization in enumerate(RUNNER.POLARIZATIONS)},
        }
        best = {
            "range_macro_relmse": 0.25,
            "range_relmse_by_polarization": {polarization: 0.25 + index * 0.05 for index, polarization in enumerate(RUNNER.POLARIZATIONS)},
        }
        report = RUNNER._optimizer_improvement(initial, best)
        self.assertAlmostEqual(report["signed_absolute_improvement"], 0.25)
        self.assertAlmostEqual(report["signed_relative_improvement"], 0.5)
        self.assertEqual(set(report["by_polarization"]), set(RUNNER.POLARIZATIONS))
        for index, polarization in enumerate(RUNNER.POLARIZATIONS):
            expected = 0.25 + index * 0.05
            self.assertAlmostEqual(report["by_polarization"][polarization]["signed_absolute_improvement"], expected)
            self.assertAlmostEqual(
                report["by_polarization"][polarization]["signed_relative_improvement"],
                expected / (0.5 + index * 0.1),
            )

    def test_bp_checkpoint_reread_reproduces_scalar_fitted_coefficients(self):
        class FakeLattice:
            def validate_ijk(self, indices):
                return np.asarray(indices, dtype=np.int64)

            def points_local(self, indices):
                return np.asarray(indices, dtype=np.float64)

            N = 3
            h_space_m = 1.0

        panel = SimpleNamespace(record_counts={polarization: 1 for polarization in RUNNER.POLARIZATIONS})
        bp = {
            polarization: {
                "coefficients_dc": np.asarray([1.0 + 2.0j]),
                "scalar_fit": {"real": 2.0, "imag": -0.5},
            }
            for polarization in RUNNER.POLARIZATIONS
        }
        class MemoryPath:
            def __init__(self):
                self.buffer = io.BytesIO()

            class _Handle:
                def __init__(self, buffer):
                    self.buffer = buffer

                def __enter__(self):
                    return self.buffer

                def __exit__(self, exc_type, exc_value, traceback):
                    self.buffer.flush()
                    return False

            def open(self, mode):
                self.buffer.seek(0)
                self.buffer.truncate(0)
                return self._Handle(self.buffer)

        path = MemoryPath()
        RUNNER._save_bp_checkpoint(path, bp, FakeLattice(), panel, np.asarray([[1, 1, 1]], dtype=np.int64))
        path.buffer.seek(0)
        with np.load(path.buffer, allow_pickle=False) as payload:
            expected = (2.0 - 0.5j) * (1.0 + 2.0j)
            self.assertEqual(payload["hh_coefficients_dc"].item(), expected)
            self.assertEqual(payload["hh_coefficients_dc_unfitted"].item(), 1.0 + 2.0j)
            np.testing.assert_array_equal(payload["hh_scalar_fit"], np.asarray([2.0, -0.5]))

    def test_torch_path_contract_is_static_when_torch_is_unavailable(self):
        core_source = (ROOT / "rift" / "gotcha_camry_fullpol_core.py").read_text(encoding="utf-8")
        core_tree = ast.parse(core_source)
        functions = {node.name: node for node in ast.walk(core_tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        forward = functions["forward_torch"]
        self.assertIn("records", [argument.arg for argument in forward.args.kwonlyargs])
        residual_source = ast.get_source_segment(core_source, functions["_projected_residuals"])
        self.assertIn("records=records", residual_source)
        energy_source = ast.get_source_segment(core_source, functions["_normalized_energy_scale"])
        self.assertIn("_normalized_coefficient_magnitude_sq", energy_source)
        self.assertIn("self.model.point_count", energy_source)
        fit_source = ast.get_source_segment(core_source, functions["fit"])
        self.assertIn("self.optimizer_updates += 1", fit_source)
        self.assertIn("self.panel.logical_batch_count", fit_source)
        runner_source = (ROOT / "scripts" / "run_gotcha_camry_fullpol_v1.py").read_text(encoding="utf-8")
        self.assertIn("model.to(execution_device)", runner_source)
        self.assertIn("preflight, shards = preflight_archives(paths)", runner_source)
        self.assertIn("trainer.load_checkpoint(latest_path)", runner_source)
        self.assertIn("bp = None", runner_source)
        self.assertIn("cgls = None", runner_source)
        self.assertIn("train_relative_errors", runner_source)
        self.assertIn("signed_relative_improvement", runner_source)
        self.assertIn("all_initialized", runner_source)
        self.assertIn("optimizer_diagnostics", runner_source)
        self.assertIn("retained_energy_fraction", core_source)
        self.assertIn("_cpu_rng_tensor", core_source)
        self.assertIn("_optimizer_step_with_diagnostics", core_source)
        runtime_validator_source = (ROOT / "scripts" / "validate_gotcha_camry_fullpol_runtime.py").read_text(encoding="utf-8")
        self.assertIn("EXPECTED_CORE_TEST_COUNT = 19", runtime_validator_source)
        self.assertIn("result.skipped", runtime_validator_source)
        self.assertIn("result.failures", runtime_validator_source)
        self.assertIn("result.errors", runtime_validator_source)


if __name__ == "__main__":
    unittest.main()
