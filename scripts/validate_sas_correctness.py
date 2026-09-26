#!/usr/bin/env python
"""Bounded CPU regressions for the contained C1 sonar correctness fixes."""

from __future__ import annotations

import json
import os
import sys
import copy
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_sas
from rift import sas_operator
from rift.sas_dataset import SASCache
from scripts import prepare_airsas_cache


torch.set_num_threads(1)


def check(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)
    print(f"PASS {label}")


def expect_raises(error_type, function, label: str) -> None:
    try:
        function()
    except error_type:
        print(f"PASS {label}")
    else:
        raise AssertionError(label)


def test_metric_quantities_and_zero_target_aggregation() -> None:
    predicted = torch.tensor([-1.0 + 0.0j])
    target = torch.tensor([1.0 + 0.0j])
    metrics = train_sas.metric_record(predicted, target)
    check(float(metrics["complex_error_sum"]) == 4.0, "complex residual numerator remains coherent")
    check(float(metrics["l1_mag_sum"]) == 0.0, "magnitude L1 is amplitude residual")
    check(float(metrics["mse_mag_sum"]) == 0.0, "magnitude MSE is amplitude residual")
    check(float(metrics["target_power"]) == 1.0, "metric target power is not clamped")

    model = torch.nn.Identity()
    calibration = torch.nn.Identity()
    cache = SimpleNamespace(num_bins=1)
    args = SimpleNamespace(eval_pings=0, eval_bins=0)
    rendered = iter((
        train_sas.metric_record(torch.tensor([1.0 + 0.0j]), torch.tensor([0.0 + 0.0j])),
        train_sas.metric_record(torch.tensor([1.0 + 0.0j]), torch.tensor([1.0 + 0.0j])),
    ))

    def fake_render(*_args, **_kwargs):
        return torch.zeros(()), next(rendered), {}

    with mock.patch.object(train_sas, "render_one", side_effect=fake_render):
        aggregate = train_sas.evaluate(model, calibration, cache, [0, 1], args, torch.device("cpu"))
    check(abs(aggregate["rel_mse"] - 1.0) < 1.0e-7, "zero-target trace contributes its prediction error")
    check("amplitude_residuals" in aggregate["metric_convention"], "evaluation records metric convention")

    zero_cache = SimpleNamespace(num_bins=1)
    zero_args = SimpleNamespace(eval_pings=0, eval_bins=0)
    zero_metrics = train_sas.metric_record(torch.tensor([1.0 + 0.0j]), torch.tensor([0.0 + 0.0j]))
    with mock.patch.object(train_sas, "render_one", return_value=(torch.zeros(()), zero_metrics, {})):
        expect_raises(
            ValueError,
            lambda: train_sas.evaluate(model, calibration, zero_cache, [0], zero_args, torch.device("cpu")),
            "all-zero target cohort is rejected as undefined",
        )


class SelectionField:
    """Differentiable renderer stub with nonconstant density and scattering."""

    def __init__(self) -> None:
        self.scale = torch.nn.Parameter(torch.tensor(0.7))

    @staticmethod
    def _density(points: torch.Tensor) -> torch.Tensor:
        return 0.15 + 0.11 * points[:, 0].square() + 0.07 * points[:, 1].square()

    def query_density(self, points: torch.Tensor, raster=None) -> torch.Tensor:
        del raster
        return self._density(points)

    def query_sas(self, points: torch.Tensor, directions: torch.Tensor, **_kwargs):
        del directions
        real = self.scale * (1.0 + 0.3 * points[:, 0] - 0.2 * points[:, 1])
        imag = self.scale * (0.2 + 0.1 * points[:, 2])
        scatterer = torch.complex(real, imag)
        normals = torch.zeros_like(points)
        normals[:, 2] = 1.0
        return {
            "density": self._density(points),
            "scatterer": scatterer,
            "normals": normals,
        }


def render_selection(field: SelectionField, bins=None) -> torch.Tensor:
    corners = torch.tensor([
        [-0.2, -0.2, 0.0], [-0.2, -0.2, 0.3], [-0.2, 0.2, 0.0], [-0.2, 0.2, 0.3],
        [0.2, -0.2, 0.0], [0.2, -0.2, 0.3], [0.2, 0.2, 0.0], [0.2, 0.2, 0.3],
    ], dtype=torch.float32)
    radii = torch.linspace(1.45, 2.05, 5)
    tx = torch.tensor([1.0, 0.11, 0.18])
    rx = torch.tensor([0.98, 0.09, 0.16])
    return sas_operator.render_sas_bins(
        field,
        radii,
        tx,
        rx,
        corners,
        num_rays=16,
        opacity_scale=3.0,
        lambertian_ratio=0.2,
        normal_step=0.01,
        output_bin_indices=bins,
    )[0]


def test_output_bin_selection_contract() -> None:
    field = SelectionField()
    full = render_selection(field)
    permutation = torch.tensor([4, 2, 0, 3, 1])
    permuted = render_selection(field, permutation)
    check(torch.allclose(permuted, full.detach()[permutation], atol=1.0e-6, rtol=1.0e-6),
          "full-length permuted selection preserves context and order")
    subset_indices = torch.tensor([4, 0, 2])
    subset = render_selection(field, subset_indices)
    check(torch.allclose(subset, full.detach()[subset_indices], atol=1.0e-6, rtol=1.0e-6),
          "short selection preserves context and order")
    sorted_full = render_selection(field, torch.arange(5))
    check(torch.allclose(sorted_full, full.detach(), atol=0.0, rtol=0.0),
          "ordered full selection preserves fast-path results")

    field.scale.grad = None
    full_for_grad = render_selection(field)
    full_loss = full_for_grad[permutation].abs().square().sum()
    full_grad = torch.autograd.grad(full_loss, field.scale)[0]
    perm_for_grad = render_selection(field, permutation)
    perm_loss = perm_for_grad.abs().square().sum()
    perm_grad = torch.autograd.grad(perm_loss, field.scale)[0]
    check(torch.allclose(perm_grad, full_grad, atol=1.0e-6, rtol=1.0e-6),
          "permuted selection gradients match indexed full output")

    expect_raises(ValueError, lambda: render_selection(field, torch.empty(0, dtype=torch.long)),
                  "empty output selection is rejected")
    expect_raises(ValueError, lambda: render_selection(field, torch.tensor([1, 1])),
                  "duplicate output selection is rejected")
    expect_raises(TypeError, lambda: render_selection(field, torch.tensor([1.5, 2.0])),
                  "non-integer output selection is rejected")


def make_split_cache(explicit: bool = True):
    return SimpleNamespace(
        has_explicit_splits=explicit,
        train_indices=np.asarray([0], dtype=np.int64),
        validation_indices=np.asarray([1], dtype=np.int64),
        test_indices=np.asarray([2], dtype=np.int64),
        num_pings=3,
        corners=np.asarray([
            [-0.2, -0.2, 0.0], [-0.2, -0.2, 0.3], [-0.2, 0.2, 0.0], [-0.2, 0.2, 0.3],
            [0.2, -0.2, 0.0], [0.2, -0.2, 0.3], [0.2, 0.2, 0.0], [0.2, 0.2, 0.3],
        ], dtype=np.float32),
        manifest={},
    )


class StopAtRestore:
    def __init__(self) -> None:
        self.context = None

    def on_restore(self, **context) -> None:
        self.context = context
        raise RuntimeError("stop after restore capture")


def run_restore_capture(cache, extra_args):
    observer = StopAtRestore()
    output_name = f"c1_restore_{os.getpid()}_{id(observer)}"
    argv = [
        "--cache", "unused", "--model", "rift_sas", "--checkpoint-name", output_name,
        "--checkpoint-root", str(PROJECT_ROOT / "tmp"), "--device", "cpu", "--steps", "0",
        "--granularity", "2", "--sh-degree", "0", *extra_args,
    ]
    with mock.patch.object(train_sas, "load_sas_cache", return_value=cache):
        try:
            train_sas.main(argv, diagnostic_observer=observer)
        except RuntimeError as exc:
            check(str(exc) == "stop after restore capture", "restore probe stopped at requested hook")
        else:
            raise AssertionError("restore probe did not stop")
    return observer


def test_resume_and_explicit_split_contract() -> None:
    cache = make_split_cache(True)
    missing = PROJECT_ROOT / "tmp" / f"missing_resume_{os.getpid()}.pt"
    missing_argv = [
        "--cache", "unused", "--model", "rift_sas", "--checkpoint-name", "c1_missing",
        "--checkpoint-root", str(PROJECT_ROOT / "tmp"), "--device", "cpu", "--steps", "0",
        "--resume", str(missing), "--granularity", "2", "--sh-degree", "0",
    ]
    with mock.patch.object(train_sas, "load_sas_cache", return_value=cache), \
         mock.patch.object(train_sas, "build_model", side_effect=AssertionError("model built before missing resume rejection")):
        for extra in ([], ["--eval-only"]):
            expect_raises(
                FileNotFoundError,
                lambda extra=extra: train_sas.main(missing_argv + extra),
                "explicit missing resume is rejected before model construction",
            )

    for require_flag in ([], ["--require-explicit-splits"]):
        observer = run_restore_capture(cache, require_flag)
        check(observer.context["train_indices"].tolist() == [0], "fixed-grid train split is preserved")
        check(observer.context["validation_indices"].tolist() == [1], "fixed-grid validation split is preserved")
        check(observer.context["test_indices"].tolist() == [2], "fixed-grid test split is preserved")

    expect_raises(
        ValueError,
        lambda: run_restore_capture(cache, ["--max-pings", "1"]),
        "explicit split cache rejects max-pings truncation",
    )
    expect_raises(
        ValueError,
        lambda: run_restore_capture(make_split_cache(False), ["--require-explicit-splits"]),
        "required explicit split rejects cache without splits",
    )

    fresh = StopAtRestore()
    fresh_name = f"c1_fresh_{os.getpid()}_{id(fresh)}"
    fresh_argv = [
        "--cache", "unused", "--model", "rift_sas", "--checkpoint-name", fresh_name,
        "--checkpoint-root", str(PROJECT_ROOT / "tmp"), "--device", "cpu", "--steps", "0",
        "--granularity", "2", "--sh-degree", "0",
    ]
    with mock.patch.object(train_sas, "load_sas_cache", return_value=make_split_cache(False)):
        try:
            train_sas.main(fresh_argv, diagnostic_observer=fresh)
        except RuntimeError as exc:
            check(str(exc) == "stop after restore capture", "fresh startup reaches restore hook")
        else:
            raise AssertionError("fresh startup probe did not stop")
    check(fresh.context["start"] == 0 and fresh.context["state"] is None,
          "absent implicit latest checkpoint remains a fresh start")


def test_normalization_metadata_contract() -> None:
    class MetadataPath:
        def __init__(self, payload):
            self.payload = payload

        @property
        def parent(self):
            return self

        def __truediv__(self, _name):
            return self

        def exists(self):
            return True

        def read_text(self, encoding=None):
            del encoding
            return json.dumps(self.payload)

        def __str__(self):
            return "synthetic_commandline_args.txt"

    def run(payload):
        return prepare_airsas_cache._normalization_contract(MetadataPath(payload), "train_only")

    mode, contract = run({
        "normalization_mode": "train_only",
        "normalization_scale": 2.0,
        "max_transmissions": 43200,
        "normalize_each": False,
    })
    check(mode == "train_only" and contract["max_transmissions"] == 43200,
          "correct train-only normalization metadata is accepted")
    expect_raises(RuntimeError, lambda: run({
        "normalization_mode": "reed_global_max",
        "normalization_scale": 2.0,
        "max_transmissions": 43200,
    }), "contradictory normalization mode is rejected")
    expect_raises(RuntimeError, lambda: run({
        "normalization_mode": "train_only",
        "normalization_scale": 2.0,
        "max_transmissions": 43200,
        "normalize_each": True,
    }), "per-waveform normalization metadata is rejected")
    expect_raises(RuntimeError, lambda: run({
        "normalization_mode": "train_only",
        "normalization_scale": 2.0,
        "max_transmissions": 43200.5,
    }), "non-integral transmission count is rejected")
    expect_raises(RuntimeError, lambda: run({
        "normalization_mode": "train_only",
        "normalization_scale": 2.0,
        "max_transmissions": True,
    }), "boolean transmission count is rejected")


def c2_manifest() -> dict:
    return {
        "complete": True,
        "cache_contract_version": 2,
        "dataset_identity": "synthetic_c2",
        "frontend_asset_identity": "synthetic_asset",
        "scene": "synthetic",
        "bandwidth_khz": 5.0,
        "num_pings": 3,
        "num_bins": 1,
        "original_num_bins": 1,
        "ring_size": 1,
        "num_rings": 3,
        "ring_protocol": "synthetic_rings",
        "held_out_protocol": "synthetic_holdout",
        "split_contract": {
            "explicit": True,
            "train_indices": [0],
            "validation_indices": [1],
            "test_indices": [2],
            "train_ring_residue": "train",
            "validation_ring_residue": "validation",
            "test_ring_residue": "test",
            "counts": {"train_rings": 1, "validation_rings": 1, "test_rings": 1},
        },
        "frontend_normalization": {
            "mode": "train_only",
            "strict_train_only": True,
            "supplied_scale": 2.0,
            "max_transmissions": 43200,
        },
        "aggressive_crop": {
            "original_min_dist": 1.0,
            "original_max_dist": 2.0,
            "new_min_sample": 0,
            "new_max_sample_exclusive": 1,
            "cropped_min_dist": 1.0,
            "cropped_max_dist": 1.0,
        },
        "crop": {"min_sample": 0, "min_dist": 1.0, "max_dist": 1.0, "num_samples": 1},
        "sound_speed_mps": 343.0,
        "sample_rate_hz": 1_000_000.0,
        "geometry_grid_shape": [2, 2, 2],
    }


def make_c2_cache(manifest=None):
    return SimpleNamespace(
        has_explicit_splits=True,
        weights=np.ones((3, 1), dtype=np.complex64),
        tx_coords=np.zeros((3, 3), dtype=np.float32),
        rx_coords=np.zeros((3, 3), dtype=np.float32),
        radii=np.asarray([1.0], dtype=np.float32),
        corners=np.asarray([
            [-0.2, -0.2, 0.0], [-0.2, -0.2, 0.3], [-0.2, 0.2, 0.0], [-0.2, 0.2, 0.3],
            [0.2, -0.2, 0.0], [0.2, -0.2, 0.3], [0.2, 0.2, 0.0], [0.2, 0.2, 0.3],
        ], dtype=np.float32),
        voxels=np.asarray([[0.0, 0.0, 0.15]], dtype=np.float32),
        source_ids=np.asarray([10, 11, 12], dtype=np.int64),
        train_indices=np.asarray([0], dtype=np.int64),
        validation_indices=np.asarray([1], dtype=np.int64),
        test_indices=np.asarray([2], dtype=np.int64),
        tx_vecs=None,
        num_pings=3,
        num_bins=1,
        manifest=copy.deepcopy(c2_manifest() if manifest is None else manifest),
    )


def make_c2_state(cache, *, step=7, best_step=7, best_metric=0.1):
    saved_argv = [
        "--cache", "saved-cache", "--model", "rift_sas", "--checkpoint-name", "saved",
        "--device", "cpu", "--steps", str(step), "--granularity", "2", "--sh-degree", "0",
        "--num-rays", "16", "--max-bins", "1", "--opacity-scale", "73", "--normal-step", "0.005",
        "--signal-scale", "3", "--beamwidth-deg", "30", "--eval-every", "5",
        "--eval-pings", "1", "--eval-bins", "0", "--checkpoint-every", "5", "--log-every", "5",
        "--seed", "11", "--calibration-mode", "legacy_cartesian",
    ]
    args = train_sas.parse_args(saved_argv)
    args.opacity_normalize = False
    model = train_sas.build_model(args, cache, torch.device("cpu"))
    calibration = train_sas.build_calibration("legacy_cartesian", torch.device("cpu"))
    optimizer = train_sas._optimizer_for_model(model, calibration, args)
    history = [{"step": best_step, "train_loss": 1.0, "train_rel_mse": 1.0, "val_rel_mse": best_metric,
                "val_l1_real": 0.0, "val_l1_imag": 0.0, "val_l1_mag": 0.0, "elapsed_seconds": 0.0}]
    if step != best_step:
        history.append({"step": step, "train_loss": 1.0, "train_rel_mse": 1.0, "val_rel_mse": 0.9,
                        "val_l1_real": 0.0, "val_l1_imag": 0.0, "val_l1_mag": 0.0, "elapsed_seconds": 0.0})
    rng = np.random.default_rng(11)
    state = train_sas.checkpoint(model, calibration, optimizer, step, best_metric, rng, history, args, cache)
    return args, model, calibration, optimizer, state


def save_state(path: Path, state) -> None:
    torch.save(state, path)


def test_missing_cache_manifests_are_rejected() -> None:
    cache = make_c2_cache({})
    sas_cache = SASCache(
        root=PROJECT_ROOT / "tmp" / "missing_manifest_cache",
        weights=cache.weights,
        tx_coords=cache.tx_coords,
        rx_coords=cache.rx_coords,
        radii=cache.radii,
        corners=cache.corners,
        voxels=cache.voxels,
        source_ids=cache.source_ids,
        train_indices=cache.train_indices,
        validation_indices=cache.validation_indices,
        test_indices=cache.test_indices,
        tx_vecs=cache.tx_vecs,
        manifest={},
    )
    expect_raises(
        ValueError,
        lambda: train_sas._validate_cache_contract({"cache_manifest": {}}, sas_cache),
        "missing manifest is rejected for SASCache",
    )
    expect_raises(
        ValueError,
        lambda: train_sas._validate_cache_contract({"cache_manifest": {}}, cache),
        "missing manifest is rejected for duck-typed cache",
    )


def c2_eval_metrics():
    return {
        "rel_mse": 4.0,
        "l1_real": 0.0,
        "l1_imag": 0.0,
        "l1_mag": 0.0,
        "mse_real": 0.0,
        "mse_imag": 0.0,
        "mse_mag": 0.0,
        "complete": 1.0,
        "views": 1.0,
        "metric_convention": "rel_mse_is_complex_residual; magnitude_fields_are_amplitude_residuals",
    }


def test_recipe_and_selected_best_contract() -> None:
    cache = make_c2_cache()
    _args, _model, _calibration, _optimizer, state = make_c2_state(cache)
    resume_path = PROJECT_ROOT / "tmp" / f"c2_resume_{os.getpid()}.pt"
    save_state(resume_path, state)
    seen = []

    def fake_evaluate(_model, _calibration, _cache, role_indices, args, _device):
        seen.append((args.signal_scale, args.opacity_scale, args.normal_step, args.beamwidth_deg,
                     args.eval_pings, args.eval_bins, role_indices.tolist()))
        return c2_eval_metrics()

    base = [
        "--cache", "moved-cache", "--model", "rift_sas", "--checkpoint-name", "c2_eval",
        "--checkpoint-root", str(PROJECT_ROOT / "tmp"), "--device", "cpu", "--eval-only",
        "--resume", str(resume_path), "--granularity", "2", "--sh-degree", "0",
    ]
    with mock.patch.object(train_sas, "load_sas_cache", return_value=cache), \
         mock.patch.object(train_sas, "evaluate", side_effect=fake_evaluate):
        train_sas.main(base)
    check(seen[-1][:4] == (3.0, 73.0, 0.005, 30.0),
          "omitted eval recipe restores saved rendering physics")
    check(seen[-1][4:6] == (1, 0), "omitted eval cohort restores saved selection")

    explicit_full = base + ["--profile", "full", "--eval-pings", "0", "--eval-bins", "0"]
    with mock.patch.object(train_sas, "load_sas_cache", return_value=cache), \
         mock.patch.object(train_sas, "evaluate", side_effect=fake_evaluate):
        train_sas.main(explicit_full)
    check(seen[-1][4:6] == (0, 0), "explicit full-role eval overrides profile eval-ping limit")

    expect_raises(
        ValueError,
        lambda: train_sas.main(base + ["--normal-step=0.006"]),
        "equals-form explicit physics mismatch is rejected",
    )
    expect_raises(
        SystemExit,
        lambda: train_sas.parse_args(base + ["--normal-st", "0.006"]),
        "abbreviated scientific flags are rejected",
    )

    expect_raises(
        ValueError,
        lambda: train_sas.main(base + ["--signal-scale", "4"]),
        "explicit physics mismatch is rejected before evaluation",
    )
    bad_manifest = c2_manifest()
    bad_manifest["dataset_identity"] = "different_dataset"
    with mock.patch.object(train_sas, "load_sas_cache", return_value=make_c2_cache(bad_manifest)):
        expect_raises(ValueError, lambda: train_sas.main(base), "different cache identity is rejected")
    missing_args_state = copy.deepcopy(state)
    del missing_args_state["args"]["signal_scale"]
    missing_path = PROJECT_ROOT / "tmp" / f"c2_missing_recipe_{os.getpid()}.pt"
    save_state(missing_path, missing_args_state)
    with mock.patch.object(train_sas, "load_sas_cache", return_value=cache):
        missing_base = list(base)
        missing_base[missing_base.index("--resume") + 1] = str(missing_path)
        expect_raises(ValueError, lambda: train_sas.main(missing_base),
                      "missing saved recipe field is reported")

    fresh_output = PROJECT_ROOT / "tmp" / f"c2_best_output_{os.getpid()}"
    with mock.patch.object(train_sas, "load_sas_cache", return_value=cache), \
         mock.patch.object(train_sas, "evaluate", return_value=c2_eval_metrics()), \
         mock.patch.object(train_sas.ComplexSHSonarField, "dense_density", return_value=torch.zeros(1)):
        train_sas.main([
            "--cache", "moved-cache", "--model", "rift_sas", "--checkpoint-name", ".",
            "--checkpoint-root", str(fresh_output), "--device", "cpu", "--steps", "7",
            "--resume", str(resume_path), "--granularity", "2", "--sh-degree", "0",
        ])
    readout = json.loads((fresh_output / "selected_readout.json").read_text(encoding="utf-8"))
    check(readout["selected_checkpoint_step"] == 7 and readout["selected_checkpoint_rel_mse"] == 0.1,
          "checkpoint that is itself best seeds a new output")

    old_best_dir = PROJECT_ROOT / "tmp" / f"c2_old_best_{os.getpid()}"
    old_best_dir.mkdir(parents=True, exist_ok=True)
    resume_old = old_best_dir / "checkpoint_latest.pt"
    _, old_model, old_cal, old_opt, old_state = make_c2_state(cache, step=7, best_step=6, best_metric=0.1)
    save_state(resume_old, old_state)
    best_state = copy.deepcopy(old_state)
    best_state["step"] = 6
    best_state["history"] = [old_state["history"][0]]
    save_state(old_best_dir / "checkpoint_best.pt", best_state)
    output_old = PROJECT_ROOT / "tmp" / f"c2_old_best_output_{os.getpid()}"
    with mock.patch.object(train_sas, "load_sas_cache", return_value=cache), \
         mock.patch.object(train_sas, "evaluate", return_value=c2_eval_metrics()), \
         mock.patch.object(train_sas.ComplexSHSonarField, "dense_density", return_value=torch.zeros(1)):
        train_sas.main([
            "--cache", "moved-cache", "--model", "rift_sas", "--checkpoint-name", ".",
            "--checkpoint-root", str(output_old), "--device", "cpu", "--steps", "7",
            "--resume", str(resume_old), "--granularity", "2", "--sh-degree", "0",
        ])
    old_readout = json.loads((output_old / "selected_readout.json").read_text(encoding="utf-8"))
    check(old_readout["selected_checkpoint_step"] == 6 and old_readout["selected_checkpoint_rel_mse"] == 0.1,
          "adjacent historical best is carried into a new output")

    missing_best_dir = PROJECT_ROOT / "tmp" / f"c2_missing_best_{os.getpid()}"
    missing_best_dir.mkdir(parents=True, exist_ok=True)
    missing_resume = missing_best_dir / "checkpoint_latest.pt"
    save_state(missing_resume, old_state)
    with mock.patch.object(train_sas, "load_sas_cache", return_value=cache), \
         mock.patch.object(train_sas, "build_model", side_effect=AssertionError("model built before missing best rejection")):
        expect_raises(
            FileNotFoundError,
            lambda: train_sas.main([
                "--cache", "moved-cache", "--model", "rift_sas", "--checkpoint-name", ".",
                "--checkpoint-root", str(PROJECT_ROOT / "tmp" / f"c2_missing_best_output_{os.getpid()}"),
                "--device", "cpu", "--steps", "7", "--resume", str(missing_resume),
                "--granularity", "2", "--sh-degree", "0",
            ]),
            "missing adjacent historical best is rejected before updates",
        )


def main() -> None:
    test_metric_quantities_and_zero_target_aggregation()
    test_output_bin_selection_contract()
    test_resume_and_explicit_split_contract()
    test_normalization_metadata_contract()
    test_missing_cache_manifests_are_rejected()
    test_recipe_and_selected_best_contract()
    print("All C1/C2 sonar correctness gates passed.")

if __name__ == "__main__":
    main()
