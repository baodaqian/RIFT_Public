"""Focused tests for the additive Camry full-polarization core."""

from __future__ import annotations

import math
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # Local authoring runtime may omit Torch.
    torch = None

if torch is not None:
    from rift.gotcha_acquisition import NativeObservationId
    from rift.gotcha_native_complex_roi_fit import CAMRY_PLACEMENT
    from rift.gotcha_camry_fullpol_core import (
        CamryCandidateLattice,
        CamryFullPolTrainer,
        CamryL3FullPolModel,
        CamryL3SparseOperator,
        CamryRangeProjector,
        CamryTrainingPanel,
        CamryTrainingRecord,
        DCCGLSResult,
        POLARIZATIONS,
        RaggedComplexValues,
        SH_BASIS_COUNT,
        _cpu_rng_tensor,
        embed_dc_coefficients,
        rank_candidate_sites,
        run_dc_cgls24,
    )
else:
    CamryCandidateLattice = None
    CamryFullPolTrainer = None
    CamryL3FullPolModel = None
    CamryL3SparseOperator = None
    CamryRangeProjector = None
    CamryTrainingPanel = None
    CamryTrainingRecord = None
    DCCGLSResult = None
    POLARIZATIONS = ("hh", "hv", "vh", "vv")
    SH_BASIS_COUNT = 16
    embed_dc_coefficients = None
    rank_candidate_sites = None
    NativeObservationId = None
    CAMRY_PLACEMENT = None
    RaggedComplexValues = None
    _cpu_rng_tensor = None


def _native_observation(
    polarization: str,
    pulse: int = 0,
    frequency_count: int = 4,
    frequency_step_hz: float | None = None,
):
    if frequency_step_hz is None:
        frequencies = np.linspace(9.0e9, 12.0e9, frequency_count, dtype=np.float64)
    else:
        frequencies = 9.0e9 + float(frequency_step_hz) * np.arange(frequency_count, dtype=np.float64)
    raw = np.linspace(1.0, 2.0, frequency_count).astype(np.complex128) + 1j * 0.25
    corrected = polarization in {"hh", "vv"}
    pulse_offset = float(pulse)
    return SimpleNamespace(
        identity=NativeObservationId(1, polarization, 2, pulse),
        role="train",
        response=raw,
        frequencies_hz=frequencies,
        r0_m=30.0 + 0.2 * pulse_offset,
        position_xyz_m=np.asarray([0.4 * pulse_offset, 0.0, 30.0 + 0.2 * pulse_offset], dtype=np.float64),
        r_correct_raw=0.25 if corrected else None,
        ph_correct_raw=0.125 if corrected else None,
        phase_reference=SimpleNamespace(
            reference_range_field="r0",
            geometry_contract="paired_monostatic_tx_equals_rx_same_observation",
            frequency_values="native_stored_exact",
        ),
        autofocus=SimpleNamespace(
            official_available=corrected,
            applied=False,
            mode="raw_channel_own_arrays_unapplied" if corrected else "official_arrays_absent",
            source_shard_id=f"pass1_{polarization}" if corrected else None,
            range_field="r_correct_raw" if corrected else None,
            phase_field="ph_correct_raw" if corrected else None,
        ),
    )


def _records(frequency_count: int = 4):
    return tuple(
        CamryTrainingRecord.from_native_observation(_native_observation(polarization, frequency_count=frequency_count))
        for polarization in POLARIZATIONS
    )


def _range_observation(polarization: str, pulse: int = 0):
    return _native_observation(polarization, pulse=pulse, frequency_count=64, frequency_step_hz=1.5e6)


def _trainer(*, unequal: bool = False):
    by_pol = {polarization: (CamryTrainingRecord.from_native_observation(_range_observation(polarization, pulse=0)),) for polarization in POLARIZATIONS}
    if unequal:
        by_pol["hh"] = tuple(
            CamryTrainingRecord.from_native_observation(_range_observation("hh", pulse=pulse))
            for pulse in (0, 1)
        )
    panel = CamryTrainingPanel(by_pol)
    lattice = CamryCandidateLattice.from_panel(panel)
    indices = np.asarray([[lattice.N // 2, lattice.N // 2, lattice.N // 2]], dtype=np.int64)
    model = CamryL3FullPolModel(lattice.points_local(indices), lattice=lattice, site_indices_ijk=indices)
    return CamryFullPolTrainer(panel, model, require_retention=False)


@unittest.skipIf(torch is None, "Torch is not installed in the local authoring runtime")
class CamryCoreContractTest(unittest.TestCase):
    def test_source_af_is_channel_owned_and_cross_pol_is_raw(self):
        records = _records()
        by_pol = {record.polarization: record for record in records}
        self.assertEqual(by_pol["hh"].representation_tag, "source_af")
        self.assertEqual(by_pol["vv"].representation_tag, "source_af")
        self.assertEqual(by_pol["hv"].representation_tag, "raw_unapplied")
        self.assertEqual(by_pol["vh"].representation_tag, "raw_unapplied")
        self.assertTrue(by_pol["hh"].source_af_applied)
        self.assertFalse(by_pol["hv"].source_af_applied)
        self.assertAlmostEqual(by_pol["hh"].r0_selected_m, 30.25)
        self.assertAlmostEqual(by_pol["hv"].r0_selected_m, 30.0)
        np.testing.assert_allclose(
            by_pol["hh"].response_selected,
            by_pol["hh"].response_raw * np.exp(1j * 0.125),
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_array_equal(by_pol["hv"].response_selected, by_pol["hv"].response_raw)

    def test_centered_implicit_lambda2_lattice_is_not_dense(self):
        lattice = CamryCandidateLattice.from_fmax(12.0e9)
        self.assertEqual(lattice.N, int(np.floor(10.0 / lattice.h_space_m)) + 1)
        self.assertEqual(lattice.candidate_count, lattice.N ** 3)
        indices = np.asarray([[0, 0, 0], [lattice.N - 1, lattice.N - 1, lattice.N - 1]], dtype=np.int64)
        points = lattice.points_local(indices)
        np.testing.assert_allclose(points[0], -points[1], rtol=0.0, atol=1e-14)
        self.assertLessEqual(np.max(np.abs(points)), 5.0)

    def test_shared_support_rank_is_deterministic_and_bounded(self):
        indices = np.asarray([[1, 0, 0], [0, 0, 0], [2, 0, 0]], dtype=np.int64)
        scores = np.asarray([1.0, 1.0, 0.5], dtype=np.float64)
        selected = rank_candidate_sites(indices, scores, max_active=2)
        np.testing.assert_array_equal(selected, np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.int64))
        with self.assertRaises(ValueError):
            rank_candidate_sites(indices, scores, max_active=SH_BASIS_COUNT * 1024)


@unittest.skipIf(torch is None, "Torch is not installed in the local authoring runtime")
class CamryCoreOperatorTest(unittest.TestCase):
    def setUp(self):
        self.records = _records()
        self.points_local = np.asarray([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.5, -0.5]], dtype=np.float64)
        self.operator = CamryL3SparseOperator(
            CAMRY_PLACEMENT.apply(self.points_local),
            point_chunk_size=2,
        )
        self.model = CamryL3FullPolModel(self.points_local, point_chunk_size=2)
        self.coefficients = np.zeros((self.operator.point_count, SH_BASIS_COUNT), dtype=np.complex128)
        self.coefficients[0, 0] = 0.2 + 0.1j
        self.coefficients[1, 5] = -0.15 + 0.03j
        self.coefficients[2, 15] = 0.04 - 0.05j

    def test_numpy_operator_matches_torch_l3_model(self):
        self.model.set_l3_coefficients({polarization: self.coefficients for polarization in POLARIZATIONS})
        torch_values = self.model.forward_channel("hh", self.records[:1], apply_gain=False)[0].detach().cpu().numpy()
        numpy_values = self.operator.forward(self.records[:1], self.coefficients).values[0]
        np.testing.assert_allclose(torch_values, numpy_values, rtol=2.0e-12, atol=2.0e-12)

    def test_dc_embedding_preserves_native_prediction(self):
        dc = np.asarray([0.3 + 0.1j, -0.2 + 0.05j, 0.04 - 0.03j], dtype=np.complex128)
        embedded = embed_dc_coefficients({polarization: dc for polarization in POLARIZATIONS}, self.operator.point_count)
        dc_prediction = self.operator.forward_dc(self.records[:1], dc).values[0]
        l3_prediction = self.operator.forward(self.records[:1], embedded["hh"]).values[0]
        np.testing.assert_allclose(dc_prediction, l3_prediction, rtol=0.0, atol=2.0e-13)


@unittest.skipIf(torch is None, "Torch is not installed in the local authoring runtime")
class CamryRangeSubsetTest(unittest.TestCase):
    def test_identity_aware_subset_projection_uses_matching_record_geometry(self):
        first = CamryTrainingRecord.from_native_observation(_range_observation("hh", pulse=0))
        second = CamryTrainingRecord.from_native_observation(_range_observation("hh", pulse=1))
        projector = CamryRangeProjector((first, second))
        frequency_count = int(first.frequencies_hz.size)
        value_first = torch.arange(frequency_count, dtype=torch.float64).to(torch.complex128)
        value_second = (torch.arange(frequency_count, dtype=torch.float64) + 1.0).to(torch.complex128)
        full = projector.forward_torch((value_first, value_second))
        subset = projector.forward_torch((value_second,), records=(second,))[0]
        np.testing.assert_allclose(subset.detach().cpu().numpy(), full[1].detach().cpu().numpy(), rtol=0.0, atol=1.0e-13)
        wrong_geometry = projector.forward_torch((value_second,), records=(first,))[0]
        self.assertFalse(torch.allclose(subset, wrong_geometry))
        adjoint_full = projector.adjoint_torch(full)
        adjoint_subset = projector.adjoint_torch((full[1],), records=(second,))[0]
        np.testing.assert_allclose(adjoint_subset.detach().cpu().numpy(), adjoint_full[1].detach().cpu().numpy(), rtol=0.0, atol=1.0e-13)

    def test_unequal_channel_counts_allow_one_record_logical_batches(self):
        records = {
            "hh": tuple(CamryTrainingRecord.from_native_observation(_range_observation("hh", pulse=pulse)) for pulse in (0, 1)),
            "hv": (CamryTrainingRecord.from_native_observation(_range_observation("hv", pulse=0)),),
            "vh": (CamryTrainingRecord.from_native_observation(_range_observation("vh", pulse=0)),),
            "vv": (CamryTrainingRecord.from_native_observation(_range_observation("vv", pulse=0)),),
        }
        panel = CamryTrainingPanel(records)
        lattice = CamryCandidateLattice.from_panel(panel)
        indices = np.asarray([[lattice.N // 2, lattice.N // 2, lattice.N // 2]], dtype=np.int64)
        model = CamryL3FullPolModel(lattice.points_local(indices), lattice=lattice, site_indices_ijk=indices)
        trainer = CamryFullPolTrainer(panel, model)
        loss = trainer.data_loss_torch(batch_index=1)
        self.assertTrue(bool(torch.isfinite(loss)))


@unittest.skipIf(torch is None, "Torch is not installed in the local authoring runtime")
class CamryTrainingContractTest(unittest.TestCase):
    class _IdentityPanel:
        def records(self, polarization):
            return (NativeObservationId(1, polarization, 2, 0),)

    class _IdentityProjector:
        def __init__(self, polarization, breakdown=False):
            self.identity = NativeObservationId(1, polarization, 2, 0)
            self.breakdown = breakdown

        def target(self):
            return RaggedComplexValues((self.identity,), (np.asarray([1.0 + 0.0j]),))

        def apply_forward(self, values):
            return values

        def apply_adjoint(self, values):
            return values

    class _IdentityOperator:
        def __init__(self, breakdown=False):
            self.point_count = 1
            self.breakdown = breakdown

        def forward_dc(self, records, coefficients):
            value = 0.0 + 0.0j if self.breakdown else complex(coefficients[0])
            return RaggedComplexValues(tuple(records), (np.asarray([value]),))

        def adjoint_dc(self, records, residuals):
            return np.asarray([residuals.values[0][0]], dtype=np.complex128)

    def test_dc_cgls_decreases_residual_and_records_early_termination(self):
        panel = self._IdentityPanel()
        projectors = {polarization: self._IdentityProjector(polarization) for polarization in POLARIZATIONS}
        result = run_dc_cgls24(panel, self._IdentityOperator(), projectors, max_iterations=24)
        for polarization in POLARIZATIONS:
            diagnostics = result.diagnostics[polarization]
            self.assertEqual(result.termination[polarization], "normal_equation_stationarity")
            self.assertEqual(diagnostics[0]["iteration"], 0)
            self.assertGreater(diagnostics[0]["residual_norm2"], diagnostics[-1]["residual_norm2"])
            self.assertEqual(diagnostics[-1]["iteration"], 1)
            self.assertEqual(diagnostics[-1]["termination"], "normal_equation_stationarity")

    def test_dc_cgls_records_nonpositive_denominator_breakdown(self):
        panel = self._IdentityPanel()
        projectors = {polarization: self._IdentityProjector(polarization, breakdown=True) for polarization in POLARIZATIONS}
        result = run_dc_cgls24(panel, self._IdentityOperator(breakdown=True), projectors, max_iterations=24)
        for polarization in POLARIZATIONS:
            self.assertEqual(result.termination[polarization], "nonpositive_search_denominator_breakdown")
            self.assertEqual(result.diagnostics[polarization][-1]["iteration"], 0)
            self.assertEqual(result.diagnostics[polarization][-1]["termination"], "nonpositive_search_denominator_breakdown")

    def test_mean_unequal_logical_batch_losses_and_gradients_match_full_macro_plus_one_prior(self):
        trainer = _trainer(unequal=True)
        trainer.Q0 = 1.0
        trainer.S_group = 1.0
        trainer.S_energy = 1.0
        with torch.no_grad():
            trainer.model.heads["hh"].w_re[0, 0] = 0.25
            trainer.model.heads["hv"].w_im[0, 1] = -0.15
        parameters = tuple(trainer.model.parameters())
        full_loss = trainer.loss_torch()
        full_gradients = torch.autograd.grad(full_loss, parameters)
        U = trainer.panel.logical_batch_count
        logical_losses = tuple(trainer.loss_torch(batch_index=index) for index in range(U))
        mean_loss = sum(logical_losses) / float(U)
        mean_gradients = torch.autograd.grad(mean_loss, parameters)
        torch.testing.assert_close(mean_loss, full_loss, rtol=2.0e-11, atol=2.0e-11)
        for expected, actual in zip(full_gradients, mean_gradients):
            torch.testing.assert_close(actual, expected, rtol=2.0e-10, atol=2.0e-10)

    def test_gain_warm_start_and_rms_normalization_preserve_predictions(self):
        trainer = _trainer()
        coefficients_dc = {polarization: np.asarray([0.2 + 0.05j], dtype=np.complex128) for polarization in POLARIZATIONS}
        trainer.model.set_l3_coefficients(embed_dc_coefficients(coefficients_dc, trainer.model.point_count))
        trainer.model.set_gains_identity()
        for polarization in POLARIZATIONS:
            records = trainer.panel.records(polarization)
            raw = trainer.model.forward_channel(polarization, records, apply_gain=False)
            projected = trainer.projectors[polarization].forward_torch(raw)
            target = trainer._torch_targets(polarization, projected[0].device)
            trainer.model.gains[polarization].maybe_init_scale(torch.cat(projected), torch.cat(target))
            self.assertTrue(bool(trainer.model.gains[polarization].initialized))
        before = {
            polarization: tuple(value.detach().clone() for value in trainer.projectors[polarization].forward_torch(trainer.model.forward_channel(polarization, trainer.panel.records(polarization))))
            for polarization in POLARIZATIONS
        }
        trainer.model.normalize_rms_one()
        after = {
            polarization: tuple(value.detach().clone() for value in trainer.projectors[polarization].forward_torch(trainer.model.forward_channel(polarization, trainer.panel.records(polarization))))
            for polarization in POLARIZATIONS
        }
        for polarization in POLARIZATIONS:
            for expected, actual in zip(before[polarization], after[polarization]):
                torch.testing.assert_close(actual, expected, rtol=2.0e-11, atol=2.0e-11)

    def test_group_and_sh_priors_are_invariant_to_head_gain_gauge_rescaling(self):
        trainer = _trainer()
        trainer.Q0 = 1.0
        trainer.S_group = 1.0
        trainer.S_energy = 1.0
        with torch.no_grad():
            for index, polarization in enumerate(POLARIZATIONS, start=1):
                trainer.model.heads[polarization].w_re.fill_(0.02 * index)
                trainer.model.heads[polarization].w_im.fill_(0.01 * index)
        base_group, base_sh = trainer.prior_terms_torch()
        base_loss = trainer.prior_loss_torch()
        with torch.no_grad():
            for index, polarization in enumerate(POLARIZATIONS, start=1):
                scale = float(index + 1)
                trainer.model.heads[polarization].w_re.mul_(scale)
                trainer.model.heads[polarization].w_im.mul_(scale)
                trainer.model.gains[polarization].log_mag.sub_(np.log(scale))
        new_group, new_sh = trainer.prior_terms_torch()
        new_loss = trainer.prior_loss_torch()
        torch.testing.assert_close(new_group, base_group, rtol=2.0e-11, atol=2.0e-11)
        torch.testing.assert_close(new_sh, base_sh, rtol=2.0e-11, atol=2.0e-11)
        torch.testing.assert_close(new_loss, base_loss, rtol=2.0e-11, atol=2.0e-11)
    def test_normalized_energy_scale_uses_each_channel_target_energy(self):
        trainer = object.__new__(CamryFullPolTrainer)
        trainer.model = SimpleNamespace(point_count=2)
        coefficients = torch.ones((4, 2, SH_BASIS_COUNT), dtype=torch.complex128)
        trainer._effective_coefficients_stack = lambda: coefficients
        trainer.target_energy = {polarization: 1.0 for polarization in POLARIZATIONS}
        unit_scale = trainer._normalized_energy_scale()
        trainer.target_energy = {"hh": 1.0, "hv": 4.0, "vh": 9.0, "vv": 16.0}
        unequal_scale = trainer._normalized_energy_scale()
        expected = (SH_BASIS_COUNT * (1.0 + 0.25 + 1.0 / 9.0 + 1.0 / 16.0))
        self.assertAlmostEqual(unequal_scale, expected, places=12)
        self.assertNotEqual(unit_scale, unequal_scale)

    def test_fit_has_exact_150_times_u_optimizer_updates(self):
        trainer = object.__new__(CamryFullPolTrainer)

        class FakeModel:
            def train(self):
                return self

            def eval(self):
                return self

        class FakeOptimizer:
            def __init__(self):
                self.steps = 0

            def zero_grad(self, set_to_none=True):
                return None

            def step(self):
                self.steps += 1

        class FakeScheduler:
            def __init__(self):
                self.steps = 0

            def step(self):
                self.steps += 1

        trainer.model = FakeModel()
        trainer.panel = SimpleNamespace(logical_batch_count=3)
        trainer.optimizer = FakeOptimizer()
        trainer.scheduler = FakeScheduler()
        trainer.history = []
        trainer.optimizer_diagnostics = []
        trainer.optimizer_updates = 0
        trainer.current_epoch = 0
        trainer.current_batch_index = 0
        trainer.best_epoch = None
        trainer.best_q = None
        trainer.loss_torch = lambda *, batch_index=None: torch.ones((), dtype=torch.float64, requires_grad=True)
        trainer.metrics = lambda: {
            "range_macro_relmse": 1.0,
            "range_relmse_by_polarization": {polarization: 1.0 for polarization in POLARIZATIONS},
        }
        result = trainer.fit(epochs=150, seed=42)
        self.assertEqual(result["optimizer_updates"], 450)
        self.assertEqual(result["best_epoch"], 1)
        self.assertEqual(trainer.optimizer.steps, 450)
        self.assertEqual(trainer.scheduler.steps, 150)
        self.assertEqual(trainer.current_epoch, 150)
        self.assertEqual(trainer.current_batch_index, 0)

    def test_checkpoint_round_trip_preserves_optimizer_scheduler_rng_and_resume_state(self):
        trainer = _trainer()
        trainer.build_optimizer()
        torch.manual_seed(1234)
        loss = trainer.loss_torch(batch_index=0)
        loss.backward()
        assert trainer.optimizer is not None
        diagnostic = trainer._optimizer_step_with_diagnostics(epoch=1)
        self.assertIsNotNone(diagnostic)
        trainer.optimizer.zero_grad(set_to_none=True)
        assert trainer.scheduler is not None
        trainer.scheduler.step()
        trainer.optimizer_updates = trainer.panel.logical_batch_count
        trainer.current_epoch = 1
        trainer.current_batch_index = 0
        trainer.history = [{"epoch": 1, "optimizer_updates": 1, "range_macro_relmse": 0.5}]
        trainer.optimizer_diagnostics = [diagnostic]
        trainer.best_epoch = 1
        trainer.best_q = 0.5
        trainer.Q0 = 0.5
        trainer.S_group = 1.0
        trainer.S_energy = 2.0
        expected_rng = torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as temporary:
            path = f"{temporary}/checkpoint_latest.pt"
            trainer.save_checkpoint(path, epoch=1, best_epoch=1, best_q=0.5, phase="latest")
            restored = _trainer()
            restored.build_optimizer()
            torch.rand(17)
            restored.load_checkpoint(path)
        for key, expected in trainer.model.state_dict().items():
            torch.testing.assert_close(restored.model.state_dict()[key], expected, rtol=0.0, atol=0.0)
        self.assertEqual(len(restored.optimizer.state), len(trainer.optimizer.state))
        for expected_state, actual_state in zip(trainer.optimizer.state.values(), restored.optimizer.state.values()):
            self.assertEqual(expected_state.keys(), actual_state.keys())
            for key in expected_state:
                if torch.is_tensor(expected_state[key]):
                    torch.testing.assert_close(actual_state[key], expected_state[key], rtol=0.0, atol=0.0)
                else:
                    self.assertEqual(actual_state[key], expected_state[key])
        self.assertEqual(restored.scheduler.state_dict(), trainer.scheduler.state_dict())
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0.0, atol=0.0)
        self.assertEqual(restored.current_epoch, 1)
        self.assertEqual(restored.current_batch_index, 0)
        self.assertEqual(restored.optimizer_updates, 1)
        self.assertEqual(restored.best_epoch, 1)
        self.assertEqual(restored.best_q, 0.5)
        self.assertEqual(restored.S_energy, 2.0)
        self.assertEqual(len(restored.optimizer_diagnostics), 1)
        self.assertEqual(restored.optimizer_diagnostics[0]["optimizer_update"], 1)

    def test_rng_state_normalization_and_malformed_checkpoint_fail_closed(self):
        source = torch.arange(32, dtype=torch.uint8)[::2]
        normalized = _cpu_rng_tensor(source, "test")
        self.assertEqual(normalized.device.type, "cpu")
        self.assertEqual(normalized.dtype, torch.uint8)
        self.assertTrue(normalized.is_contiguous())
        torch.testing.assert_close(normalized, source.contiguous())

        trainer = _trainer()
        trainer.build_optimizer()
        with tempfile.TemporaryDirectory() as temporary:
            valid_path = f"{temporary}/checkpoint.pt"
            bad_path = f"{temporary}/checkpoint_bad.pt"
            trainer.save_checkpoint(valid_path, epoch=0, best_epoch=None, best_q=None, phase="initialization")
            try:
                payload = torch.load(valid_path, map_location="cpu", weights_only=False)
            except TypeError:
                payload = torch.load(valid_path, map_location="cpu")
            payload["rng_state"] = torch.zeros((2, 2), dtype=torch.uint8)
            torch.save(payload, bad_path)
            restored = _trainer()
            restored.build_optimizer()
            with self.assertRaises(ValueError):
                restored.load_checkpoint(bad_path)

    def test_optimizer_step_diagnostics_match_exact_group_statistics(self):
        trainer = _trainer()
        trainer.build_optimizer(scene_lr=1.0e-2, gain_lr=2.0e-2)
        with torch.no_grad():
            for index, polarization in enumerate(POLARIZATIONS, start=1):
                trainer.model.heads[polarization].w_re.fill_(0.1 * index)
                trainer.model.heads[polarization].w_im.fill_(0.05 * index)
        assert trainer.optimizer is not None
        trainer.optimizer.zero_grad(set_to_none=True)
        trainer.loss_torch(batch_index=0).backward()
        group = next(group for group in trainer.optimizer.param_groups if group["name"] == "head_hh")
        parameters = tuple(group["params"])
        expected_gradient_rms = math.sqrt(
            sum(float(torch.sum(torch.abs(parameter.grad.detach()) ** 2).item()) for parameter in parameters)
            / sum(int(parameter.numel()) for parameter in parameters)
        )
        expected_parameter_rms = math.sqrt(
            sum(float(torch.sum(torch.abs(parameter.detach()) ** 2).item()) for parameter in parameters)
            / sum(int(parameter.numel()) for parameter in parameters)
        )
        before = tuple(parameter.detach().clone() for parameter in parameters)
        diagnostic = trainer._optimizer_step_with_diagnostics(epoch=1)
        assert diagnostic is not None
        observed = diagnostic["groups"]["head_hh"]
        update_sum_square = sum(
            float(torch.sum(torch.abs(parameter.detach() - snapshot) ** 2).item())
            for parameter, snapshot in zip(parameters, before)
        )
        expected_update_rms = math.sqrt(update_sum_square / sum(int(parameter.numel()) for parameter in parameters))
        expected_relative_step = math.sqrt(update_sum_square) / max(
            math.sqrt(sum(float(torch.sum(torch.abs(snapshot) ** 2).item()) for snapshot in before)),
            np.finfo(np.float64).tiny,
        )
        self.assertAlmostEqual(observed["gradient_rms"], expected_gradient_rms, places=13)
        self.assertAlmostEqual(observed["parameter_rms_before"], expected_parameter_rms, places=13)
        self.assertAlmostEqual(observed["lr_used"], 1.0e-2, places=15)
        self.assertAlmostEqual(observed["update_rms"], expected_update_rms, places=13)
        self.assertAlmostEqual(observed["relative_step"], expected_relative_step, places=13)
        self.assertEqual({name for name in diagnostic["groups"]}, {f"{kind}_{polarization}" for polarization in POLARIZATIONS for kind in ("head", "gain")})

    def test_optimizer_diagnostics_propagate_to_all_150_history_rows(self):
        trainer = object.__new__(CamryFullPolTrainer)
        parameters = [torch.nn.Parameter(torch.ones(1, dtype=torch.float64)) for _ in range(8)]

        class FakeModel:
            def train(self):
                return self

            def eval(self):
                return self

        class FakeOptimizer:
            def __init__(self):
                self.steps = 0
                names = [f"{kind}_{polarization}" for polarization in POLARIZATIONS for kind in ("head", "gain")]
                self.param_groups = [
                    {"params": [parameter], "lr": 1.0e-2 if index % 2 == 0 else 2.0e-2, "name": name}
                    for index, (parameter, name) in enumerate(zip(parameters, names))
                ]

            def zero_grad(self, set_to_none=True):
                for group in self.param_groups:
                    for parameter in group["params"]:
                        parameter.grad = None

            def step(self):
                self.steps += 1
                for group in self.param_groups:
                    for parameter in group["params"]:
                        if parameter.grad is not None:
                            parameter.data.sub_(group["lr"] * parameter.grad)

        class FakeScheduler:
            def __init__(self):
                self.steps = 0

            def step(self):
                self.steps += 1

        trainer.model = FakeModel()
        trainer.panel = SimpleNamespace(logical_batch_count=3)
        trainer.optimizer = FakeOptimizer()
        trainer.scheduler = FakeScheduler()
        trainer.history = []
        trainer.optimizer_diagnostics = []
        trainer.optimizer_updates = 0
        trainer.current_epoch = 0
        trainer.current_batch_index = 0
        trainer.best_epoch = None
        trainer.best_q = None
        trainer.loss_torch = lambda *, batch_index=None: sum(parameter * 0.0 for parameter in parameters)
        trainer.metrics = lambda: {
            "range_macro_relmse": 1.0,
            "range_relmse_by_polarization": {polarization: 1.0 for polarization in POLARIZATIONS},
        }
        result = trainer.fit(epochs=150, seed=42)
        self.assertEqual(len(result["history"]), 150)
        self.assertEqual(len(trainer.optimizer_diagnostics), 450)
        for row in result["history"]:
            self.assertEqual(set(row["optimizer_diagnostics"]["groups"]), {f"{kind}_{polarization}" for polarization in POLARIZATIONS for kind in ("head", "gain")})
            self.assertEqual(row["optimizer_diagnostics"]["scene"]["group_count"], 4)
            self.assertEqual(row["optimizer_diagnostics"]["gain"]["group_count"], 4)

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA is required for the production checkpoint contract")
    def test_cuda_checkpoint_round_trip_restores_rng_optimizer_devices_and_resume_readiness(self):
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        trainer = _trainer()
        trainer.model.to(device)
        trainer.build_optimizer()
        torch.manual_seed(701)
        torch.cuda.manual_seed_all(702)
        trainer.optimizer.zero_grad(set_to_none=True)
        trainer.loss_torch(batch_index=0).backward()
        diagnostic = trainer._optimizer_step_with_diagnostics(epoch=1)
        trainer.optimizer.zero_grad(set_to_none=True)
        trainer.optimizer_updates = trainer.panel.logical_batch_count
        trainer.current_epoch = 1
        trainer.current_batch_index = 0
        trainer.history = [{"epoch": 1, "optimizer_updates": 1}]
        trainer.optimizer_diagnostics = [diagnostic]
        trainer.best_epoch = 1
        trainer.best_q = 0.5
        assert trainer.scheduler is not None
        trainer.scheduler.step()
        expected_cpu_rng = torch.get_rng_state().clone()
        expected_cuda_rng = tuple(state.clone() for state in torch.cuda.get_rng_state_all())
        with tempfile.TemporaryDirectory() as temporary:
            best_path = f"{temporary}/checkpoint_best.pt"
            latest_path = f"{temporary}/checkpoint_latest.pt"
            trainer.save_checkpoint(best_path, epoch=1, best_epoch=1, best_q=0.5, phase="best")
            trainer.save_checkpoint(latest_path, epoch=1, best_epoch=1, best_q=0.5, phase="latest")
            current_cuda = torch.cuda.current_device()
            for checkpoint_path in (best_path, latest_path):
                restored = _trainer()
                restored.model.to(device)
                restored.build_optimizer()
                torch.rand(17)
                torch.rand(17, device=device)
                restored.load_checkpoint(checkpoint_path)
                torch.testing.assert_close(torch.get_rng_state(), expected_cpu_rng, rtol=0.0, atol=0.0)
                for actual, expected in zip(torch.cuda.get_rng_state_all(), expected_cuda_rng):
                    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
                expected_cpu_generator = torch.Generator(device="cpu")
                expected_cpu_generator.set_state(expected_cpu_rng)
                torch.testing.assert_close(
                    torch.rand(9, generator=expected_cpu_generator),
                    torch.rand(9),
                    rtol=0.0,
                    atol=0.0,
                )
                expected_cuda_generator = torch.Generator(device=device)
                expected_cuda_generator.set_state(expected_cuda_rng[current_cuda])
                torch.testing.assert_close(
                    torch.rand(9, device=device, generator=expected_cuda_generator),
                    torch.rand(9, device=device),
                    rtol=0.0,
                    atol=0.0,
                )
                for parameter, state in restored.optimizer.state.items():
                    group = next(group for group in restored.optimizer.param_groups if any(parameter is candidate for candidate in group["params"]))
                    for key, value in state.items():
                        if not torch.is_tensor(value):
                            continue
                        expected_device = torch.device("cpu") if key == "step" else parameter.device
                        if bool(group.get("capturable", False)) or bool(group.get("fused", False)):
                            expected_device = parameter.device if key == "step" else parameter.device
                        self.assertEqual(value.device, expected_device)
                restored.optimizer.zero_grad(set_to_none=True)
                loss = restored.loss_torch(batch_index=0)
                self.assertTrue(bool(torch.isfinite(loss)))
                loss.backward()
                post_load_diagnostic = restored._optimizer_step_with_diagnostics(epoch=2)
                self.assertIsNotNone(post_load_diagnostic)
                restored.optimizer.zero_grad(set_to_none=True)
                self.assertEqual(restored.current_epoch, 1)
                self.assertEqual(restored.current_batch_index, 0)
                self.assertEqual(restored.optimizer_updates, 1)
                self.assertEqual(len(restored.optimizer_diagnostics), 1)


if __name__ == "__main__":
    unittest.main()
