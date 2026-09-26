#!/usr/bin/env python
"""CPU contract checks for the scoped sonar calibration repair."""

from __future__ import annotations

import io
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

torch.set_num_threads(1)

import train_sas
from train_sas import (
    ComplexCalibration,
    LogPolarCalibration,
    _optimizer_for_model,
    build_calibration,
    calibration_diagnostics,
    checkpoint,
    evaluate,
    parse_args,
    resolve_calibration_mode,
)


def clone_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


def assert_tensor_mapping_equal(test: unittest.TestCase, first: dict, second: dict) -> None:
    test.assertEqual(set(first), set(second))
    for key in first:
        test.assertTrue(torch.equal(first[key], second[key]), key)


def assert_nested_equal(first, second) -> None:
    if isinstance(first, torch.Tensor) or isinstance(second, torch.Tensor):
        if not isinstance(first, torch.Tensor) or not isinstance(second, torch.Tensor):
            raise AssertionError("optimizer state tensor mismatch")
        if not torch.equal(first, second):
            raise AssertionError("optimizer state tensor values differ")
        return
    if isinstance(first, dict) or isinstance(second, dict):
        if not isinstance(first, dict) or not isinstance(second, dict):
            raise AssertionError("optimizer state mapping mismatch")
        if set(first) != set(second):
            raise AssertionError("optimizer state keys differ")
        for key in first:
            assert_nested_equal(first[key], second[key])
        return
    if isinstance(first, (list, tuple)) or isinstance(second, (list, tuple)):
        if type(first) is not type(second) or len(first) != len(second):
            raise AssertionError("optimizer state sequence mismatch")
        for left, right in zip(first, second):
            assert_nested_equal(left, right)
        return
    if first != second:
        raise AssertionError(f"optimizer state values differ: {first!r} != {second!r}")


def tiny_model() -> torch.nn.Linear:
    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 0.3], [-0.4, 0.7]]))
    return model


def complex_prediction(model: torch.nn.Linear, inputs: torch.Tensor) -> torch.Tensor:
    values = model(inputs)
    return torch.complex(values[:, 0], values[:, 1])


def update_once(
    model: torch.nn.Linear,
    calibration: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    target: torch.Tensor,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    predicted = calibration(complex_prediction(model, inputs))
    loss = torch.nn.functional.mse_loss(predicted.real, target.real) + torch.nn.functional.mse_loss(
        predicted.imag, target.imag
    )
    loss.backward()
    optimizer.step()


class CalibrationContractTests(unittest.TestCase):
    def test_01_known_gain(self) -> None:
        raw = torch.tensor([1.0 + 1.0j, 2.0 - 1.0j], dtype=torch.complex64)
        gain = torch.polar(torch.tensor(2.0), torch.tensor(0.7))
        target = gain * raw
        calibration = LogPolarCalibration()
        calibration.maybe_initialize(raw, target)
        self.assertTrue(bool(calibration.initialized))
        self.assertTrue(torch.allclose(calibration(raw), target, rtol=1e-5, atol=1e-6))
        before = clone_state(calibration)
        calibration.maybe_initialize(raw, target + 4.0 - 2.0j)
        assert_tensor_mapping_equal(self, before, clone_state(calibration))
        print("PASS 1 known gain")

    def test_02_orthogonal_warm_start(self) -> None:
        raw_values = torch.tensor([1.0, 1.0], dtype=torch.float32)
        target = torch.tensor([1.0 - 0.0j, -1.0 + 0.0j], dtype=torch.complex64)
        for mode in ("legacy_cartesian", "log_polar"):
            raw_field = torch.nn.Parameter(raw_values.clone())
            raw = torch.complex(raw_field, torch.zeros_like(raw_field))
            calibration = build_calibration(mode, torch.device("cpu"))
            calibration.maybe_initialize(raw, target)
            predicted = calibration(raw)
            loss = torch.nn.functional.mse_loss(predicted.real, target.real) + torch.nn.functional.mse_loss(
                predicted.imag, target.imag
            )
            loss.backward()
            if mode == "legacy_cartesian":
                self.assertEqual(float(calibration.real), 0.0)
                self.assertTrue(torch.equal(raw_field.grad, torch.zeros_like(raw_field)))
            else:
                self.assertTrue(torch.allclose(calibration(torch.ones((), dtype=torch.complex64)).abs(), torch.ones(()), atol=1e-6))
                self.assertTrue(torch.isfinite(raw_field.grad).all())
                self.assertGreater(float(raw_field.grad.norm()), 0.0)
        print("PASS 2 orthogonal warm start")

    def test_03_zero_then_nonzero_prediction(self) -> None:
        calibration = LogPolarCalibration()
        zero = torch.zeros(2, dtype=torch.complex64)
        target = torch.tensor([1.0 - 0.0j, -1.0 + 0.0j], dtype=torch.complex64)
        before = clone_state(calibration)
        calibration.maybe_initialize(zero, target)
        self.assertFalse(bool(calibration.initialized))
        assert_tensor_mapping_equal(self, before, clone_state(calibration))
        calibration.maybe_initialize(torch.ones(2, dtype=torch.complex64), target)
        self.assertTrue(bool(calibration.initialized))
        self.assertTrue(torch.allclose(calibration(torch.ones((), dtype=torch.complex64)).abs(), torch.ones(()), atol=1e-6))
        print("PASS 3 zero then nonzero prediction")

    def test_04_adam_relative_scale(self) -> None:
        ratios = {}
        for mode in ("legacy_cartesian", "log_polar"):
            calibration = build_calibration(mode, torch.device("cpu"))
            with torch.no_grad():
                if mode == "legacy_cartesian":
                    calibration.real.fill_(6.0e-5)
                    calibration.imag.zero_()
                else:
                    calibration.log_mag.fill_(math.log(6.0e-5))
                    calibration.phase.zero_()
                calibration.initialized.fill_(True)
            optimizer = torch.optim.Adam(calibration.parameters(), lr=1.0e-3)
            one = torch.ones((), dtype=torch.complex64)
            old_magnitude = float(calibration(one).abs())
            loss = (calibration(one).real - 1.0).square()
            loss.backward()
            optimizer.step()
            ratios[mode] = float(calibration(one).abs()) / old_magnitude
        self.assertGreater(ratios["legacy_cartesian"], 10.0)
        self.assertGreaterEqual(ratios["log_polar"], 1.0009)
        self.assertLessEqual(ratios["log_polar"], 1.0011)
        print("PASS 4 Adam relative scale")

    def test_05_mode_resolution(self) -> None:
        self.assertEqual(resolve_calibration_mode("adaptive_rift_sas", "auto"), "log_polar")
        self.assertEqual(resolve_calibration_mode("rift_sas", "auto"), "legacy_cartesian")
        self.assertEqual(resolve_calibration_mode("sh_sas", "auto"), "legacy_cartesian")
        self.assertEqual(resolve_calibration_mode("adaptive_rift_sas", "log_polar"), "log_polar")
        self.assertEqual(resolve_calibration_mode("adaptive_rift_sas", "legacy_cartesian"), "legacy_cartesian")
        old_state = {"calibration_state_dict": ComplexCalibration().state_dict()}
        new_state = {"calibration_state_dict": LogPolarCalibration().state_dict()}
        self.assertEqual(resolve_calibration_mode("adaptive_rift_sas", "auto", old_state), "legacy_cartesian")
        self.assertEqual(resolve_calibration_mode("adaptive_rift_sas", "auto", new_state), "log_polar")
        with self.assertRaisesRegex(ValueError, "retain its calibration parameterization"):
            resolve_calibration_mode("adaptive_rift_sas", "log_polar", old_state)
        with self.assertRaisesRegex(ValueError, "metadata/state disagreement"):
            resolve_calibration_mode(
                "adaptive_rift_sas", "auto",
                {**old_state, "calibration_mode": "log_polar"},
            )
        with self.assertRaisesRegex(ValueError, "unsupported calibration state keys"):
            resolve_calibration_mode(
                "adaptive_rift_sas", "auto",
                {"calibration_state_dict": {"wrong": torch.tensor(0)}},
            )
        self.assertIsInstance(build_calibration("legacy_cartesian", torch.device("cpu")), ComplexCalibration)
        self.assertIsInstance(build_calibration("log_polar", torch.device("cpu")), LogPolarCalibration)
        print("PASS 5 mode resolution table")

    def test_06_state_optimizer_round_trip(self) -> None:
        inputs = torch.eye(2, dtype=torch.float32)
        target = torch.tensor([0.3 + 0.2j, -0.1 + 0.5j], dtype=torch.complex64)
        for mode in ("legacy_cartesian", "log_polar"):
            args = SimpleNamespace(model="adaptive_rift_sas", calibration_mode=mode, lr=1.0e-3)
            cache = SimpleNamespace(manifest={})
            model = tiny_model()
            calibration = build_calibration(mode, torch.device("cpu"))
            optimizer = _optimizer_for_model(model, calibration, args)
            calibration.maybe_initialize(complex_prediction(model, inputs), target)
            for _ in range(3):
                update_once(model, calibration, optimizer, inputs, target)
            rng = np.random.default_rng(42)
            payload = checkpoint(model, calibration, optimizer, 3, 1.0, rng, [], args, cache)
            self.assertEqual(payload["calibration_mode"], mode)
            self.assertEqual(payload["args"]["calibration_mode"], mode)
            buffer = io.BytesIO()
            torch.save(payload, buffer)
            buffer.seek(0)
            restored_state = torch.load(buffer, map_location="cpu", weights_only=False)
            if mode == "legacy_cartesian":
                restored_state.pop("calibration_mode")
                restored_state["args"] = dict(restored_state["args"])
                restored_state["args"].pop("calibration_mode")
            restored_mode = resolve_calibration_mode("adaptive_rift_sas", "auto", restored_state)
            self.assertEqual(restored_mode, mode)
            restored_model = tiny_model()
            restored_calibration = build_calibration(restored_mode, torch.device("cpu"))
            restored_optimizer = _optimizer_for_model(restored_model, restored_calibration, args)
            restored_model.load_state_dict(restored_state["model_state_dict"])
            restored_calibration.load_state_dict(restored_state["calibration_state_dict"])
            restored_optimizer.load_state_dict(restored_state["optimizer_state_dict"])
            before_original = calibration(complex_prediction(model, inputs))
            before_restored = restored_calibration(complex_prediction(restored_model, inputs))
            self.assertTrue(torch.equal(before_original, before_restored))
            update_once(model, calibration, optimizer, inputs, target)
            update_once(restored_model, restored_calibration, restored_optimizer, inputs, target)
            assert_tensor_mapping_equal(self, model.state_dict(), restored_model.state_dict())
            assert_tensor_mapping_equal(self, calibration.state_dict(), restored_calibration.state_dict())
            assert_nested_equal(optimizer.state_dict(), restored_optimizer.state_dict())
        print("PASS 6 state/optimizer round trip")

    def test_07_evaluation_never_initializes_or_updates_gain(self) -> None:
        cache = SimpleNamespace(
            radii=np.asarray([1.0, 2.0], dtype=np.float32),
            tx_coords=np.zeros((1, 3), dtype=np.float32),
            rx_coords=np.zeros((1, 3), dtype=np.float32),
            corners=np.zeros((8, 3), dtype=np.float32),
            tx_vecs=None,
            weights=np.asarray([[1.0 + 0.0j, -1.0 + 0.0j]], dtype=np.complex64),
            num_bins=2,
        )
        args = parse_args([
            "--cache", "unused", "--model", "adaptive_rift_sas", "--checkpoint-name", "unused",
            "--device", "cpu", "--eval-pings", "0", "--eval-bins", "0",
        ])
        args.opacity_normalize = False

        def fake_render(*_args, **_kwargs):
            return torch.tensor([1.0 + 1.0j, 2.0 - 1.0j], dtype=torch.complex64), {}

        for mode in ("legacy_cartesian", "log_polar"):
            for initialized in (False, True):
                model = tiny_model()
                calibration = build_calibration(mode, torch.device("cpu"))
                if initialized:
                    with torch.no_grad():
                        if mode == "legacy_cartesian":
                            calibration.real.fill_(0.25)
                            calibration.imag.fill_(-0.1)
                        else:
                            calibration.log_mag.fill_(math.log(0.25))
                            calibration.phase.fill_(-0.1)
                        calibration.initialized.fill_(True)
                before = clone_state(calibration)
                with mock.patch.object(train_sas, "render_sas_bins", side_effect=fake_render):
                    metrics = evaluate(model, calibration, cache, [0], args, torch.device("cpu"))
                assert_tensor_mapping_equal(self, before, clone_state(calibration))
                self.assertEqual(metrics["views"], 1.0)
        print("PASS 7 evaluation leaves gain unchanged")

    def test_08_diagnostics_are_detached_and_exact(self) -> None:
        calibration = LogPolarCalibration()
        with torch.no_grad():
            calibration.initialized.fill_(True)
        raw = torch.tensor([1.0 + 1.0j, 2.0 - 1.0j], dtype=torch.complex64)
        aux = {
            "calibration_raw": raw.detach(),
            "calibration_target": raw.detach(),
            "calibration_predicted": raw.detach(),
            "transmittance": torch.tensor([[1.0, 0.5], [0.0, 0.0001]], dtype=torch.float32),
            "lambertian": torch.tensor([0.0, 1.0, 2.0, 0.0], dtype=torch.float32),
        }
        state_before = clone_state(calibration)
        rng_before = torch.get_rng_state().clone()
        values = calibration_diagnostics(aux, calibration)
        expected_keys = [
            "raw_rms", "target_rms", "pred_rms", "raw_corr_abs", "gain_abs", "gain_phase",
            "T_all_min", "T_all_mean", "T_all_lt_1e3", "lambert_positive",
        ]
        self.assertEqual(list(values), expected_keys)
        self.assertTrue(all(isinstance(value, float) for value in values.values()))
        for key in ("raw_rms", "target_rms", "pred_rms"):
            self.assertAlmostEqual(values[key], math.sqrt(3.5), places=6)
        self.assertAlmostEqual(values["raw_corr_abs"], 1.0, places=6)
        self.assertAlmostEqual(values["gain_abs"], 1.0, places=6)
        self.assertAlmostEqual(values["gain_phase"], 0.0, places=6)
        self.assertAlmostEqual(values["T_all_min"], 0.0, places=6)
        self.assertAlmostEqual(values["T_all_mean"], 0.375025, places=6)
        self.assertAlmostEqual(values["T_all_lt_1e3"], 0.5, places=6)
        self.assertAlmostEqual(values["lambert_positive"], 0.5, places=6)
        zero_aux = dict(aux)
        zero_aux["calibration_raw"] = torch.zeros_like(raw)
        zero_values = calibration_diagnostics(zero_aux, calibration)
        self.assertTrue(math.isnan(zero_values["raw_corr_abs"]))
        assert_tensor_mapping_equal(self, state_before, clone_state(calibration))
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        print("PASS 8 diagnostics are detached and exact")


if __name__ == "__main__":
    unittest.main(verbosity=2)
