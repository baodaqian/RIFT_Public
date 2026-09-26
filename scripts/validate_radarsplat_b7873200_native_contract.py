#!/usr/bin/env python
"""Data-free native renderer, optimization, and continuation tests for RadarSplat B7873200.

Run this on an allocated PACE node with PyTorch.  It does not open B787 data,
does not use a cache, and never evaluates a sealed response role.
"""

from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import tempfile
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train_radarsplat as trainer
from rift.radarsplat_b7873200 import (
    RadarSplatEffects,
    RadarSplatGrid,
    RadarSplatModel,
    cartesian_to_spherical_gaussians,
)
from rift.radarsplat_b7873200_adapter import (
    OCCUPANCY_THRESHOLD,
    PRUNE_OPACITY,
    ObjectiveWeights,
    balanced_occupancy_l1,
    create_optimizers,
    export_gaussian_occupancy_geometry,
    load_native_view,
    model_from_checkpoint_state,
    native_power_from_matched_filter,
    native_power_objective,
    native_ssim_index,
    occupancy_mask,
    prune_low_opacity,
    target_concentration,
    target_grid_from_arrays,
)
from rift.radarsplat_b7873200_acquisition import acquisition_payload
from rift.radarsplat_b7873200_acquisition import _expected_target_geometry, write_acquisition_record
from rift.radarsplat_b7873200_protocol import (
    ACQUISITION_FILENAME,
    CACHE_SCHEMA,
    MANIFEST_FILENAME,
    RECIPE_FILENAME,
    STATS_FILENAME,
    TARGET_DIRECTORY,
    TARGET_SCHEMA,
    atomic_save_npz,
    atomic_write_json,
    b7873200_sealed_identity,
    expected_cache_recipe,
    load_cache,
    target_path,
)


class Gates:
    def __init__(self) -> None:
        self.count = 0

    def check(self, condition: bool, message: str) -> None:
        if not condition:
            raise AssertionError(message)
        self.count += 1
        print(f"PASS {self.count:02d}: {message}", flush=True)


def _acquisition_record(indices: tuple[int, ...] = (3, 5, 7, 9)) -> dict[str, np.ndarray]:
    """Small direct-calibration fixture with the real archive header semantics."""

    count = len(indices)
    return acquisition_payload(
        view_indices=np.asarray(indices, dtype=np.int64),
        frequency_hz=np.linspace(8.5e9, 11.5e9, 600, dtype=np.float64),
        viewpoint_positions=np.stack(
            [np.asarray((10.0 + number, 0.0, 0.0), dtype=np.float64) for number in range(count)]
        ),
        tx_pos=np.tile(np.linspace(-0.01, 0.01, 16, dtype=np.float64)[None, :, None], (count, 1, 3)),
        rx_pos=np.tile(np.linspace(-0.01, 0.01, 16, dtype=np.float64)[None, :, None], (count, 1, 3)),
        scene_center_m=np.zeros(3, dtype=np.float64),
        metadata={"radar_fc_hz": 10.0e9, "radar_bandwidth_hz": 3.0e9},
        response_shape=(10_000, 16, 16, 1, 600),
        response_dtype="complex64",
    )


def _grid() -> RadarSplatGrid:
    return RadarSplatGrid(
        num_range_bins=12,
        range_resolution_m=0.1,
        range_start_m=0.45,
        azimuth_start_deg=-6.0,
        azimuth_span_deg=12.0,
        output_azimuth_resolution_deg=0.5,
        intermediate_azimuth_resolution_deg=0.5,
        azimuth_beamwidth_deg=0.5,
        # The released filter truncates sigma to integer pixels.  At this
        # 0.1 m fixture resolution, 0.7 m yields a genuine one-pixel sigma.
        spectral_leakage_width_m=0.7,
    )


def _effects() -> RadarSplatEffects:
    return RadarSplatEffects(
        use_noise_probability=True,
        use_spectral_leakage=False,
        use_azimuth_antenna_gain=False,
        use_multipath=False,
        output_floor=1.0e-7,
        output_ceiling=1.0,
    )


def _model(mean: tuple[float, float, float], *, opacity: float = 0.35) -> RadarSplatModel:
    return RadarSplatModel(
        torch.tensor([mean], dtype=torch.float32),
        initial_scale=0.08,
        initial_opacity=opacity,
        initial_noise_probability=0.05,
        initial_reflectance=torch.tensor([0.55]),
        sh_degree=1,
        seed=17,
    )


def _assert_state_equal(left: object, right: object, label: str) -> None:
    if torch.is_tensor(left) and torch.is_tensor(right):
        if left.dtype.is_floating_point or left.dtype.is_complex:
            if not torch.equal(left, right):
                raise AssertionError(f"tensor state differs at {label}")
        elif not torch.equal(left, right):
            raise AssertionError(f"tensor state differs at {label}")
        return
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not (isinstance(left, np.ndarray) and isinstance(right, np.ndarray)):
            raise AssertionError(f"array/non-array state differs at {label}")
        if left.shape != right.shape or left.dtype != right.dtype or not np.array_equal(left, right):
            raise AssertionError(f"array state differs at {label}")
        return
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            raise AssertionError(f"mapping keys differ at {label}")
        for key in left:
            _assert_state_equal(left[key], right[key], f"{label}.{key}")
        return
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            raise AssertionError(f"sequence length differs at {label}")
        for number, (a, b) in enumerate(zip(left, right)):
            _assert_state_equal(a, b, f"{label}[{number}]")
        return
    if left != right:
        raise AssertionError(f"state differs at {label}: {left!r} != {right!r}")


def _test_direct_comparator(gates: Gates) -> None:
    gates.check(
        trainer._directly_equal({"a": [torch.tensor([1.0])]}, {"a": [torch.tensor([1.0])]}),
        "direct continuation comparator accepts exactly equal nested state",
    )
    gates.check(
        not trainer._directly_equal(True, 1)
        and not trainer._directly_equal([1, 2], (1, 2))
        and not trainer._directly_equal(torch.tensor([1]), np.asarray([1])),
        "direct continuation comparator rejects loose scalar, sequence, and tensor/array matches",
    )
    _assert_state_equal(
        {"frequency_hz": np.asarray([8.5e9, 11.5e9], dtype=np.float64)},
        {"frequency_hz": np.asarray([8.5e9, 11.5e9], dtype=np.float64)},
        "equal calibration array",
    )
    try:
        _assert_state_equal(
            np.asarray([8.5e9, 11.5e9], dtype=np.float64),
            np.asarray([8.5e9, 11.5e9 + 1.0], dtype=np.float64),
            "changed calibration array",
        )
    except AssertionError:
        gates.check(True, "contract comparator handles equal and changed NumPy calibration arrays directly")
    else:
        raise AssertionError("changed calibration array must fail direct test comparison")


def _test_target_projection_and_grid(gates: Gates) -> None:
    amplitude = torch.tensor(
        [1 + 2j, 2 - 1j, 3 + 0j, 2 + 0j, 1j, -1 + 1j], dtype=torch.complex64
    )
    observed = native_power_from_matched_filter(
        amplitude, n_elevation=2, n_azimuth=1, n_range=3
    )
    expected = torch.tensor([[9.0, 6.0, 11.0]])
    gates.check(torch.equal(observed, expected), "native target is exactly elevation-summed squared MF magnitude")
    try:
        native_power_from_matched_filter(torch.ones(6), n_elevation=2, n_azimuth=1, n_range=3)
    except TypeError:
        gates.check(True, "native target projection rejects a fabricated real/coherent substitute")
    else:
        raise AssertionError("native target projection must require complex MF amplitude")

    axes = {
        "range_m": np.asarray([0.95, 1.05, 1.15], dtype=np.float32),
        "azimuth_rad": np.deg2rad(np.asarray([-0.9, 0.0, 0.9], dtype=np.float64)).astype(np.float32),
    }
    spec = {
        "scene_extent_m": 0.15,
        "scene_center_m": [0.0, 0.0, 0.0],
        "azimuth_center_deg": 0.0,
        "n_range": 3,
        "n_azimuth": 3,
        "n_elevation": 2,
        "output_azimuth_resolution_deg": 0.9,
        "intermediate_azimuth_resolution_deg": 0.09,
        "azimuth_beamwidth_deg": 1.8,
        # With 10 cm range bins, the release's integer sigma formula needs at
        # least 60 cm.  Keep this direct-grid fixture executable, not merely
        # parseable.
        "spectral_leakage_width_m": 0.7,
    }
    grid = target_grid_from_arrays(**axes, expected_grid=spec)
    gates.check(
        math.isclose(grid.range_start_m, 0.9, abs_tol=2e-6)
        and math.isclose(grid.azimuth_start_deg, -1.35, abs_tol=2e-5),
        "stored target centres map to half-bin renderer edges",
    )
    gates.check(
        grid.output_azimuth_bins == 3
        and grid.intermediate_azimuth_bins == 30
        and grid.azimuth_stride == 10
        and math.isclose(grid.intermediate_azimuth_start_offset_deg, 0.405, abs_tol=1e-12),
        "output/intermediate azimuth bins retain the Q=10 centred convention",
    )
    undersized_leakage = dict(spec)
    undersized_leakage["spectral_leakage_width_m"] = 0.2
    try:
        target_grid_from_arrays(**axes, expected_grid=undersized_leakage)
    except ValueError:
        gates.check(True, "target-grid adapter rejects a release leakage width that quantizes to zero pixels")
    else:
        raise AssertionError("zero-pixel spectral leakage must be rejected before rendering")
    b787_spec = {
        **spec,
        "n_range": 32,
        "n_azimuth": 32,
    }
    b787_axes = {
        "range_m": (
            (10.0 - 0.15)
            + (np.arange(32, dtype=np.float64) + 0.5) * (0.3 / 32.0)
        ).astype(np.float32),
        "azimuth_rad": np.deg2rad(
            (-0.5 * 32 + 0.5 + np.arange(32, dtype=np.float64)) * 0.9
        ).astype(np.float32),
    }
    b787_grid = target_grid_from_arrays(**b787_axes, expected_grid=b787_spec)
    gates.check(
        math.isclose(b787_grid.range_resolution_m, 0.009375, rel_tol=0.0, abs_tol=0.0)
        and b787_grid.output_azimuth_bins == 32
        and b787_grid.intermediate_azimuth_bins == 320
        and b787_grid.azimuth_stride == 10
        and math.isclose(b787_grid.azimuth_span_deg, 28.8, rel_tol=0.0, abs_tol=1.0e-12)
        and math.isclose(b787_grid.azimuth_start_deg, -14.4, rel_tol=0.0, abs_tol=1.0e-12),
        "B787 float32 10 m centres reconstruct exact declared 32-bin and Q=10 grid physics",
    )
    bad_axes = dict(b787_axes)
    bad_axes["range_m"] = b787_axes["range_m"].copy()
    bad_axes["range_m"][7] += np.float32(1.0e-4)
    try:
        target_grid_from_arrays(**bad_axes, expected_grid=b787_spec)
    except ValueError:
        gates.check(True, "declared physical lattice rejects a range centre beyond float32 quantization")
    else:
        raise AssertionError("perturbed B787 range axis must be rejected")


def _test_sensor_transform(gates: Gates) -> None:
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
    )
    pose = torch.eye(4, dtype=torch.float64)
    pose[:3, :3] = rotation
    means = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
    covariance = torch.diag(torch.tensor([4.0, 9.0, 16.0], dtype=torch.float64)).unsqueeze(0)
    result = cartesian_to_spherical_gaussians(means, covariance, pose)
    expected_sensor_covariance = rotation.T @ covariance[0] @ rotation
    gates.check(
        torch.allclose(result["means_sensor"][0, 0], torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))
        and torch.allclose(result["covariances_sensor"][0, 0], expected_sensor_covariance),
        "projection rotates world covariance into the calibrated sensor frame",
    )
    gates.check(
        torch.allclose(result["means_spherical"][0, 0], torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)),
        "rotated boresight target lands at range one, azimuth zero, elevation zero",
    )


def _test_normalization_and_balance(gates: Gates) -> None:
    train = torch.tensor([[0.25, 0.5]], dtype=torch.float32)
    validation = torch.tensor([[2.0]], dtype=torch.float32)
    peak = float(train.max())
    gates.check(
        torch.equal(train / peak, torch.tensor([[0.5, 1.0]])) and float(validation / peak) == 4.0,
        "train-only peak leaves a larger validation value unclipped",
    )
    target = torch.tensor([[0.0, OCCUPANCY_THRESHOLD, 0.0, 0.0]], dtype=torch.float32)
    mask = occupancy_mask(target)
    predicted = torch.tensor([[0.1, 0.7, 0.2, 0.4]], dtype=torch.float32)
    observed, mode = balanced_occupancy_l1(predicted, mask)
    expected = 0.5 * abs(0.7 - 1.0) + 0.5 * np.mean([0.1, 0.2, 0.4])
    gates.check(mode == "balanced" and math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=1e-7), "occupancy loss gives positive and background pixels equal class weight at 0.001")
    localized = torch.zeros((32, 32), dtype=torch.float32)
    localized[15, 16] = 1.0
    diffuse = torch.full((32, 32), 2.0 * OCCUPANCY_THRESHOLD)
    local_stats = target_concentration(localized)
    diffuse_stats = target_concentration(diffuse)
    gates.check(
        float(local_stats["effective_support_bins"]) < float(diffuse_stats["effective_support_bins"])
        and float(local_stats["positive_fraction"]) < float(diffuse_stats["positive_fraction"]),
        "target-concentration diagnostics distinguish localized from diffuse support without changing resolution",
    )


def _test_native_ssim(gates: Gates) -> None:
    zero = torch.zeros((8, 8), dtype=torch.float32, requires_grad=True)
    zero_ssim = native_ssim_index(zero, torch.zeros((8, 8), dtype=torch.float32))
    (1.0 - zero_ssim).backward()
    gates.check(
        math.isclose(float(zero_ssim.detach()), 1.0, rel_tol=0.0, abs_tol=1.0e-6)
        and zero.grad is not None
        and bool(torch.isfinite(zero.grad).all()),
        "native SSIM scores identical dark power as one with finite gradients",
    )
    for amplitude, label in ((1.0e-5, "low-power"), (4.0, "bright")):
        target = torch.full((8, 8), amplitude, dtype=torch.float32)
        prediction = torch.full((8, 8), 0.8 * amplitude, dtype=torch.float32, requires_grad=True)
        score = native_ssim_index(prediction, target)
        (1.0 - score).backward()
        gates.check(
            bool(torch.isfinite(score))
            and prediction.grad is not None
            and bool(torch.isfinite(prediction.grad).all()),
            f"native SSIM has finite {label} power and gradient behavior",
        )


def _test_renderer_and_learnability(gates: Gates) -> None:
    torch.manual_seed(3)
    grid = _grid()
    effects = _effects()
    pose = torch.eye(4, dtype=torch.float32)
    reference = _model((1.0, 0.0, 0.0), opacity=0.45)
    target = reference.render(pose, grid, effects, active_sh_degree=0)["final_power"].detach()
    learner = _model((0.92, 0.012, 0.0), opacity=0.22)
    initial = learner.render(pose, grid, effects, active_sh_degree=0)
    required = {"final_power", "occupancy", "clean_power", "noise_power", "gaussian_scene_power"}
    gates.check(required.issubset(initial) and not any("phase" in key for key in initial), "native renderer exposes power and occupancy products but no coherent phase output")
    loss = F.mse_loss(initial["final_power"], target)
    loss.backward()
    gradient_norms = {
        name: float(parameter.grad.abs().sum())
        for name, parameter in learner.named_parameters()
        if parameter.grad is not None
    }
    gates.check(all(math.isfinite(value) for value in gradient_norms.values()) and any(value > 0.0 for value in gradient_norms.values()), "native renderer has finite nonzero Gaussian parameter gradients")
    optimizer = torch.optim.Adam(learner.parameters(), lr=0.02)
    baseline = float(loss.detach())
    for _ in range(12):
        optimizer.zero_grad(set_to_none=True)
        observed = learner.render(pose, grid, effects, active_sh_degree=0)["final_power"]
        current = F.mse_loss(observed, target)
        current.backward()
        optimizer.step()
    final = float(F.mse_loss(learner.render(pose, grid, effects, active_sh_degree=0)["final_power"], target).detach())
    gates.check(final < baseline, "a perturbed Gaussian scene can reduce native power loss through real optimization")
    objective_render = learner.render(pose, grid, effects, active_sh_degree=1)
    for parameter in learner.parameters():
        parameter.grad = None
    objective = native_power_objective(
        objective_render,
        target[0],
        learner,
        max_scale=0.2,
        weights=ObjectiveWeights(ssim=0.2, occupancy=1.0, max_size=0.0, opacity_noise=0.0),
    )
    objective_total = objective["total"]
    assert torch.is_tensor(objective_total)
    objective_total.backward()
    gates.check(
        bool(torch.isfinite(objective_total))
        and any(parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0 for parameter in learner.parameters()),
        "native real-power objective accepts [1,A,R] renderer output and backpropagates finite nonzero gradients",
    )

    unclipped = RadarSplatModel(
        torch.tensor([[1.0, 0.0, 0.0]] * 8, dtype=torch.float32),
        initial_scale=0.08,
        initial_opacity=0.9,
        initial_noise_probability=0.1,
        initial_reflectance=torch.full((8, 3), 0.99),
        sh_degree=0,
        seed=29,
    )
    b787_render = unclipped.render(pose, grid, RadarSplatEffects.b787_clean(), active_sh_degree=0)
    b787_target = b787_render["final_power"].detach() * 1.1
    b787_loss = F.mse_loss(b787_render["final_power"], b787_target)
    b787_loss.backward()
    gates.check(
        float(b787_render["final_power"].max()) > 1.0
        and bool(torch.isfinite(b787_loss))
        and unclipped.sh0.grad is not None
        and float(unclipped.sh0.grad.abs().sum()) > 0.0,
        "unclipped B787 train-peak normalization can render and fit a target above one",
    )


def _test_prune_and_export(gates: Gates, root: Path) -> None:
    model = RadarSplatModel.random_scene(
        num_gaussians=3, extent=0.2, initial_scale=0.05,
        initial_opacity=0.3, initial_noise_probability=0.1, sh_degree=1, seed=5,
    )
    rates = {name: 1.0e-3 for name in ("means", "log_scales", "quaternions", "opacity_logits", "noise_probability_logits", "sh0", "shN")}
    optimizers = create_optimizers(model, rates)
    loss = sum(parameter.square().sum() for parameter in model.parameters())
    loss.backward()
    for optimizer in optimizers.values():
        optimizer.step()
    old_moments = optimizers["means"].state[model.means]["exp_avg"].detach().clone()
    with torch.no_grad():
        boundary_logits = torch.logit(torch.tensor([0.0004, 0.0005, 0.0008], dtype=torch.float32))
        model.opacity_logits.copy_(boundary_logits)
    gates.check(
        abs(float(model.opacity[1]) - PRUNE_OPACITY) <= 8.0 * torch.finfo(torch.float32).eps * PRUNE_OPACITY,
        "sigmoid(logit(0.0005)) is treated as the numerical prune boundary",
    )
    result = prune_low_opacity(model, optimizers, PRUNE_OPACITY)
    new_moments = optimizers["means"].state[model.means]["exp_avg"]
    gates.check(
        result["before"] == 3 and result["after"] == 2 and not bool(result["fallback_kept"])
        and torch.equal(new_moments, old_moments[1:]),
        "strict less-than 0.0005 pruning retains the boundary row and preserves Adam moments",
    )
    all_low = RadarSplatModel.random_scene(
        num_gaussians=2, extent=0.2, initial_scale=0.05,
        initial_opacity=0.1, initial_noise_probability=0.1, sh_degree=0, seed=9,
    )
    all_low_optimizers = create_optimizers(all_low, {name: 1.0e-3 for name in rates})
    result = prune_low_opacity(all_low, all_low_optimizers, threshold=0.2)
    gates.check(result["after"] == 1 and bool(result["fallback_kept"]), "pruning can never delete every Gaussian")
    export_path = export_gaussian_occupancy_geometry(
        root / "geometry.npz", model, active_sh_degree=1, train_peak_power=2.0
    )
    with np.load(export_path, allow_pickle=False) as archive:
        fields = set(archive.files)
        gates.check(
            {"means", "scales", "quaternions", "occupancy", "noise_probability"}.issubset(fields)
            and not any("phase" in field for field in fields),
            "geometry export contains native Gaussian/occupancy rows and no phase field",
        )


def _test_checkpoint_restore(gates: Gates) -> None:
    torch.manual_seed(27)
    random.seed(27)
    model = _model((1.0, 0.0, 0.0))
    rates = {name: 1.0e-3 for name in ("means", "log_scales", "quaternions", "opacity_logits", "noise_probability_logits", "sh0", "shN")}
    optimizers = create_optimizers(model, rates)
    loss = sum(parameter.square().sum() for parameter in model.parameters())
    loss.backward()
    for optimizer in optimizers.values():
        optimizer.step()
    sampler = trainer.DeterministicViewSampler((3, 5, 7), 42)
    sampler.next()
    identity = {
        "fixture": "native-continuation",
        "roles": {"train": [3, 5, 7], "validation": [9]},
        "optimization": {"steps": 2},
    }
    payload = trainer._checkpoint_payload(
        identity=identity,
        model=model,
        optimizers=optimizers,
        sampler=sampler,
        step=1,
        history=[{"step": 1, "relative_mse_native_power": 0.25}],
        best_validation_rel_mse=0.25,
        complete=False,
        last_train={"step": 1},
        acquisition_record=_acquisition_record(),
    )
    restored = model_from_checkpoint_state(payload["model_state_dict"], device="cpu", seed=27)
    restored_optimizers = create_optimizers(restored, rates)
    restored_sampler = trainer.DeterministicViewSampler((3, 5, 7), 42)
    result = trainer._restore_checkpoint(
        copy.deepcopy(payload), identity, restored, restored_optimizers, restored_sampler, _acquisition_record()
    )
    gates.check(result[0] == 1 and result[2] == 0.25 and result[4] is False, "checkpoint restores direct control state")
    _assert_state_equal(model.state_dict(), restored.state_dict(), "model")
    _assert_state_equal(
        {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
        {name: optimizer.state_dict() for name, optimizer in restored_optimizers.items()},
        "optimizer",
    )
    gates.check(sampler.state_dict() == restored_sampler.state_dict(), "checkpoint restores deterministic train-view sampler")
    try:
        trainer._restore_checkpoint(copy.deepcopy(payload), {"fixture": "changed"}, restored, restored_optimizers, restored_sampler, _acquisition_record())
    except ValueError:
        gates.check(True, "checkpoint rejects a changed direct run identity without a digest gate")
    else:
        raise AssertionError("checkpoint must reject changed scientific identity")

    nonfinite_model_payload = copy.deepcopy(payload)
    nonfinite_model_payload["model_state_dict"]["means"][0, 0] = float("nan")
    try:
        model_from_checkpoint_state(nonfinite_model_payload["model_state_dict"], device="cpu", seed=27)
    except ValueError:
        gates.check(True, "checkpoint reconstruction rejects non-finite model state before any update")
    else:
        raise AssertionError("non-finite checkpoint model state must be rejected")
    nonfloat32_model_payload = copy.deepcopy(payload)
    nonfloat32_model_payload["model_state_dict"]["means"] = (
        nonfloat32_model_payload["model_state_dict"]["means"].to(torch.float64)
    )
    try:
        model_from_checkpoint_state(nonfloat32_model_payload["model_state_dict"], device="cpu", seed=27)
    except ValueError:
        gates.check(True, "checkpoint reconstruction rejects a silently-castable non-float32 model state")
    else:
        raise AssertionError("non-float32 checkpoint model state must be rejected before continuation")
    nonfinite_adam_payload = copy.deepcopy(payload)
    means_states = nonfinite_adam_payload["optimizer_state_dicts"]["means"]["state"]
    assert means_states
    next(iter(means_states.values()))["exp_avg"][0, 0] = float("nan")
    finite_model = model_from_checkpoint_state(payload["model_state_dict"], device="cpu", seed=27)
    finite_optimizers = create_optimizers(finite_model, rates)
    finite_sampler = trainer.DeterministicViewSampler((3, 5, 7), 42)
    try:
        trainer._restore_checkpoint(
            nonfinite_adam_payload,
            identity,
            finite_model,
            finite_optimizers,
            finite_sampler,
            _acquisition_record(),
        )
    except ValueError:
        gates.check(True, "checkpoint restore rejects non-finite Adam state before the next update")
    else:
        raise AssertionError("non-finite Adam checkpoint state must be rejected")
    changed_adam_recipe = copy.deepcopy(payload)
    changed_adam_recipe["optimizer_state_dicts"]["means"]["param_groups"][0]["lr"] *= 2.0
    changed_recipe_model = model_from_checkpoint_state(payload["model_state_dict"], device="cpu", seed=27)
    changed_recipe_optimizers = create_optimizers(changed_recipe_model, rates)
    try:
        trainer._restore_checkpoint(
            changed_adam_recipe,
            identity,
            changed_recipe_model,
            changed_recipe_optimizers,
            trainer.DeterministicViewSampler((3, 5, 7), 42),
            _acquisition_record(),
        )
    except ValueError:
        gates.check(True, "checkpoint restore rejects an Adam learning-rate recipe change before an update")
    else:
        raise AssertionError("checkpoint must not replace its declared Adam recipe")
    changed_adam_weight_decay = copy.deepcopy(payload)
    changed_adam_weight_decay["optimizer_state_dicts"]["means"]["param_groups"][0]["weight_decay"] = 0.1
    changed_weight_decay_model = model_from_checkpoint_state(payload["model_state_dict"], device="cpu", seed=27)
    try:
        trainer._restore_checkpoint(
            changed_adam_weight_decay,
            identity,
            changed_weight_decay_model,
            create_optimizers(changed_weight_decay_model, rates),
            trainer.DeterministicViewSampler((3, 5, 7), 42),
            _acquisition_record(),
        )
    except ValueError:
        gates.check(True, "checkpoint restore rejects altered native Adam weight decay before an update")
    else:
        raise AssertionError("checkpoint must not accept altered native Adam semantics")
    empty_adam_payload = copy.deepcopy(payload)
    for optimizer_state in empty_adam_payload["optimizer_state_dicts"].values():
        optimizer_state["state"] = {}
    empty_adam_model = model_from_checkpoint_state(payload["model_state_dict"], device="cpu", seed=27)
    try:
        trainer._restore_checkpoint(
            empty_adam_payload,
            identity,
            empty_adam_model,
            create_optimizers(empty_adam_model, rates),
            trainer.DeterministicViewSampler((3, 5, 7), 42),
            _acquisition_record(),
        )
    except ValueError:
        gates.check(True, "positive-step checkpoint rejects a silently reset Adam state")
    else:
        raise AssertionError("positive-step checkpoint must retain at least one native Adam state")
    fractional_adam_payload = copy.deepcopy(payload)
    fractional_entry = next(iter(fractional_adam_payload["optimizer_state_dicts"]["means"]["state"].values()))
    fractional_entry["step"] = torch.tensor(1.5, dtype=torch.float32)
    fractional_adam_model = model_from_checkpoint_state(payload["model_state_dict"], device="cpu", seed=27)
    try:
        trainer._restore_checkpoint(
            fractional_adam_payload,
            identity,
            fractional_adam_model,
            create_optimizers(fractional_adam_model, rates),
            trainer.DeterministicViewSampler((3, 5, 7), 42),
            _acquisition_record(),
        )
    except ValueError:
        gates.check(True, "checkpoint restore rejects a fractional Adam progress counter")
    else:
        raise AssertionError("fractional Adam progress must be rejected")
    negative_second_moment_payload = copy.deepcopy(payload)
    negative_second_moment_entry = next(iter(negative_second_moment_payload["optimizer_state_dicts"]["means"]["state"].values()))
    negative_second_moment_entry["exp_avg_sq"][0, 0] = -1.0
    negative_second_moment_model = model_from_checkpoint_state(payload["model_state_dict"], device="cpu", seed=27)
    try:
        trainer._restore_checkpoint(
            negative_second_moment_payload,
            identity,
            negative_second_moment_model,
            create_optimizers(negative_second_moment_model, rates),
            trainer.DeterministicViewSampler((3, 5, 7), 42),
            _acquisition_record(),
        )
    except ValueError:
        gates.check(True, "checkpoint restore rejects a negative Adam second moment")
    else:
        raise AssertionError("negative Adam second moment must be rejected")
    nonfloat32_moment_payload = copy.deepcopy(payload)
    nonfloat32_moment_entry = next(iter(nonfloat32_moment_payload["optimizer_state_dicts"]["means"]["state"].values()))
    nonfloat32_moment_entry["exp_avg"] = nonfloat32_moment_entry["exp_avg"].to(torch.float64)
    nonfloat32_moment_model = model_from_checkpoint_state(payload["model_state_dict"], device="cpu", seed=27)
    try:
        trainer._restore_checkpoint(
            nonfloat32_moment_payload,
            identity,
            nonfloat32_moment_model,
            create_optimizers(nonfloat32_moment_model, rates),
            trainer.DeterministicViewSampler((3, 5, 7), 42),
            _acquisition_record(),
        )
    except ValueError:
        gates.check(True, "checkpoint restore rejects a raw Adam moment that would otherwise be cast")
    else:
        raise AssertionError("checkpoint must not silently cast raw Adam moments")
    try:
        trainer._checkpoint_payload(
            identity=identity,
            model=model,
            optimizers=optimizers,
            sampler=sampler,
            step=2,
            history=[{"step": 2, "relative_mse_native_power": float("nan")}],
            best_validation_rel_mse=float("nan"),
            complete=True,
            last_train={"step": 2},
            acquisition_record=_acquisition_record(),
        )
    except ValueError:
        gates.check(True, "checkpoint save rejects a non-finite terminal validation metric")
    else:
        raise AssertionError("non-finite terminal validation metrics must never be serialized")
    try:
        trainer._checkpoint_payload(
            identity=identity,
            model=model,
            optimizers=optimizers,
            sampler=sampler,
            step=2,
            history=[{"step": 2, "relative_mse_native_power": 0.5}],
            best_validation_rel_mse=0.25,
            complete=True,
            last_train={"step": 2},
            acquisition_record=_acquisition_record(),
        )
    except ValueError:
        gates.check(True, "terminal checkpoint save rejects a best metric that disagrees with history")
    else:
        raise AssertionError("terminal best metric must agree with the saved validation history")
    try:
        trainer._checkpoint_payload(
            identity=identity,
            model=model,
            optimizers=optimizers,
            sampler=sampler,
            step=1,
            history=[{"step": 1, "relative_mse_native_power": 0.25}],
            best_validation_rel_mse=0.25,
            complete=False,
            last_train={"step": 1, "total": float("nan")},
            acquisition_record=_acquisition_record(),
        )
    except ValueError:
        gates.check(True, "checkpoint save rejects non-finite training diagnostics")
    else:
        raise AssertionError("non-finite training diagnostics must never be serialized")
    underflow_model = _model((1.0, 0.0, 0.0))
    underflow_optimizers = create_optimizers(underflow_model, rates)
    with torch.no_grad():
        underflow_model.log_scales.fill_(-1000.0)
    try:
        trainer._assert_finite_training_state(
            underflow_model,
            underflow_optimizers,
            label="underflow fixture",
        )
    except ValueError:
        gates.check(True, "post-update/save state rejects finite log-scale values whose physical scale underflows")
    else:
        raise AssertionError("underflowed physical scales must be rejected before checkpoint save")

    terminal_without_final_validation = copy.deepcopy(payload)
    terminal_without_final_validation["step"] = 2
    terminal_without_final_validation["last_train"] = {"step": 2}
    terminal_model = model_from_checkpoint_state(
        terminal_without_final_validation["model_state_dict"], device="cpu", seed=27
    )
    terminal_optimizers = create_optimizers(terminal_model, rates)
    terminal_sampler = trainer.DeterministicViewSampler((3, 5, 7), 42)
    try:
        trainer._restore_checkpoint(
            terminal_without_final_validation,
            identity,
            terminal_model,
            terminal_optimizers,
            terminal_sampler,
            _acquisition_record(),
        )
    except ValueError:
        gates.check(True, "final-step checkpoint without a final validation row is rejected")
    else:
        raise AssertionError("final-step checkpoint without final validation must be rejected")

    pruned = RadarSplatModel.random_scene(
        num_gaussians=3, extent=0.2, initial_scale=0.05,
        initial_opacity=0.3, initial_noise_probability=0.1, sh_degree=1, seed=31,
    )
    pruned_optimizers = create_optimizers(pruned, rates)
    prime = sum(parameter.square().sum() for parameter in pruned.parameters())
    prime.backward()
    for optimizer in pruned_optimizers.values():
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        pruned.opacity_logits.copy_(torch.logit(torch.tensor([0.0004, 0.0005, 0.0008])))
    prune_low_opacity(pruned, pruned_optimizers, PRUNE_OPACITY)
    pruned_identity = {**identity, "optimization": {"steps": 3}}
    pruned_payload = trainer._checkpoint_payload(
        identity=pruned_identity,
        model=pruned,
        optimizers=pruned_optimizers,
        sampler=sampler,
        step=2,
        history=[{"step": 1, "relative_mse_native_power": 0.25}],
        best_validation_rel_mse=0.25,
        complete=False,
        last_train={"step": 2, "pruned": True},
        acquisition_record=_acquisition_record(),
    )
    resumed = model_from_checkpoint_state(pruned_payload["model_state_dict"], device="cpu", seed=31)
    resumed_optimizers = create_optimizers(resumed, rates)
    resumed_sampler = trainer.DeterministicViewSampler((3, 5, 7), 42)
    trainer._restore_checkpoint(
        copy.deepcopy(pruned_payload), pruned_identity, resumed, resumed_optimizers, resumed_sampler, _acquisition_record()
    )
    for active_model, active_optimizers in ((pruned, pruned_optimizers), (resumed, resumed_optimizers)):
        continuation_loss = sum(parameter.square().sum() for parameter in active_model.parameters())
        continuation_loss.backward()
        for optimizer in active_optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    _assert_state_equal(pruned.state_dict(), resumed.state_dict(), "pruned next-update model")
    _assert_state_equal(
        {name: optimizer.state_dict() for name, optimizer in pruned_optimizers.items()},
        {name: optimizer.state_dict() for name, optimizer in resumed_optimizers.items()},
        "pruned next-update optimizer",
    )
    gates.check(pruned.num_gaussians == 2 and resumed.num_gaussians == 2, "pruned checkpoint reconstructs topology before the next Adam update")

    inactive = RadarSplatModel.random_scene(
        num_gaussians=3, extent=0.2, initial_scale=0.05,
        initial_opacity=0.3, initial_noise_probability=0.1, sh_degree=1, seed=37,
    )
    inactive_optimizers = create_optimizers(inactive, rates)
    # This is the normal Adam behavior for a temporarily inactive branch: the
    # optimizer is stepped with the others but owns no moments until a gradient
    # reaches its parameter.  Pruning must preserve that slot as empty, not
    # turn a valid continuation into an unverifiable checkpoint.
    inactive_loss = inactive.means.square().sum() + inactive.opacity_logits.square().sum()
    inactive_loss.backward()
    for optimizer in inactive_optimizers.values():
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        inactive.opacity_logits.copy_(torch.logit(torch.tensor([0.0004, 0.0005, 0.0008])))
    prune_low_opacity(inactive, inactive_optimizers, PRUNE_OPACITY)
    inactive_identity = {**identity, "optimization": {"steps": 2}}
    inactive_payload = trainer._checkpoint_payload(
        identity=inactive_identity,
        model=inactive,
        optimizers=inactive_optimizers,
        sampler=sampler,
        step=1,
        history=[{"step": 1, "relative_mse_native_power": 0.25}],
        best_validation_rel_mse=0.25,
        complete=False,
        last_train={"step": 1, "pruned": True},
        acquisition_record=_acquisition_record(),
    )
    inactive_serialized = inactive_payload["optimizer_state_dicts"]
    gates.check(
        any(
            any(not entry for entry in state["state"].values())
            for state in inactive_serialized.values()
        ),
        "pruning preserves an explicitly empty Adam slot for an inactive parameter",
    )
    inactive_resumed = model_from_checkpoint_state(inactive_payload["model_state_dict"], device="cpu", seed=37)
    inactive_resumed_optimizers = create_optimizers(inactive_resumed, rates)
    trainer._restore_checkpoint(
        copy.deepcopy(inactive_payload),
        inactive_identity,
        inactive_resumed,
        inactive_resumed_optimizers,
        trainer.DeterministicViewSampler((3, 5, 7), 42),
        _acquisition_record(),
    )
    for active_model, active_optimizers in (
        (inactive, inactive_optimizers),
        (inactive_resumed, inactive_resumed_optimizers),
    ):
        next_loss = active_model.means.square().sum() + active_model.opacity_logits.square().sum()
        next_loss.backward()
        for optimizer in active_optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    _assert_state_equal(inactive.state_dict(), inactive_resumed.state_dict(), "inactive-pruned next-update model")
    _assert_state_equal(
        {name: optimizer.state_dict() for name, optimizer in inactive_optimizers.items()},
        {name: optimizer.state_dict() for name, optimizer in inactive_resumed_optimizers.items()},
        "inactive-pruned next-update optimizer",
    )
    gates.check(
        inactive.num_gaussians == 2 and inactive_resumed.num_gaussians == 2,
        "pruned continuation restores empty inactive Adam slots without changing topology",
    )


def _fixture_identity() -> dict[str, object]:
    permutation = np.random.Generator(np.random.PCG64(42)).permutation(10_000)
    return b7873200_sealed_identity(
        {
            "schema": "rift_npz_sealed_protocol_v1",
            "version": 1,
            "data_format": "npz",
            "response_shape": [10_000, 16, 16, 1, 600],
            "response_dtype": "complex64",
            "role_manifest_name": "b78710k_interp_seed42_train3200_val1000_test1000_v1",
            "split_strategy": "fixed_tail_subsampled",
            "role_ids": {
                "train": [int(value) for value in permutation[:3200]],
                "validation": [int(value) for value in permutation[9000:]],
                "reserved_test": [int(value) for value in permutation[8000:9000]],
                "unused": [int(value) for value in permutation[3200:8000]],
            },
            "response_access": {
                "train_materialized": True,
                "validation_materialized": True,
                "reserved_test_materialized": False,
                "unused_materialized": False,
            },
        }
    )


def _fixture_target_spec() -> dict[str, object]:
    return {
        "grid": {
            "scene_extent_m": 0.15,
            "scene_center_m": [0.0, 0.0, 0.0],
            "azimuth_center_deg": 0.0,
            "n_azimuth": 32,
            "n_elevation": 32,
            "n_range": 32,
            "output_azimuth_resolution_deg": 0.9,
            "elevation_sampling_resolution_deg": 0.9,
            "intermediate_azimuth_resolution_deg": 0.09,
            "azimuth_beamwidth_deg": 0.9,
            # The release uses integer sigma pixels.  0.06 m over the
            # 9.375 mm synthetic range lattice gives one real sigma pixel,
            # retaining a 32x32 Q=10 filtered native renderer gate.
            "spectral_leakage_width_m": 0.06,
        },
        "matched_filter": {
            "phase_sign": -1.0,
            "response_layout": "tx_rx_freq",
            "range_model": "none",
            "include_four_pi": False,
            "backend": "direct",
            "compute_dtype": "float64",
        },
        "occupancy_threshold": OCCUPANCY_THRESHOLD,
    }


def _fixture_record(indices: tuple[int, int]) -> dict[str, np.ndarray]:
    count = len(indices)
    tx = np.zeros((count, 16, 3), dtype=np.float64)
    rx = np.zeros((count, 16, 3), dtype=np.float64)
    line = np.linspace(-0.01, 0.01, 16, dtype=np.float64)
    tx[:, :, 1] = line
    rx[:, :, 2] = line
    return acquisition_payload(
        view_indices=np.asarray(indices, dtype=np.int64),
        frequency_hz=np.linspace(8.5e9, 11.5e9, 600, dtype=np.float64),
        viewpoint_positions=np.asarray(((10.0, 0.0, 0.0), (10.0, 0.05, 0.0)), dtype=np.float64),
        tx_pos=tx,
        rx_pos=rx,
        scene_center_m=np.zeros(3, dtype=np.float64),
        metadata={"radar_fc_hz": 10.0e9, "radar_bandwidth_hz": 3.0e9},
        response_shape=(10_000, 16, 16, 1, 600),
        response_dtype="complex64",
    )


def _write_fixture_target(
    root: Path,
    *,
    view_index: int,
    role: str,
    record: Mapping[str, np.ndarray],
    grid_spec: Mapping[str, object],
    power: np.ndarray,
) -> dict[str, np.ndarray]:
    pose, range_axis, azimuth_axis, elevation_axis = _expected_target_geometry(
        record, view_index, grid_spec
    )
    payload = {
        "schema": np.asarray(TARGET_SCHEMA),
        "view_index": np.asarray(view_index, dtype=np.int64),
        "role": np.asarray(role),
        "radarsplat_mf_power": np.asarray(power, dtype=np.float32),
        "sensor_to_world": pose,
        "range_m": range_axis,
        "azimuth_rad": azimuth_axis,
        "elevation_rad": elevation_axis,
    }
    atomic_save_npz(target_path(root, view_index), **payload)
    return payload


def _make_synthetic_cache(root: Path) -> tuple[dict[str, object], dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    identity = _fixture_identity()
    roles = identity["role_ids"]
    assert isinstance(roles, Mapping)
    materialized = {"train": [int(roles["train"][0])], "validation": [int(roles["validation"][0])]}
    recipe = expected_cache_recipe(identity, _fixture_target_spec(), materialized_roles=materialized)
    grid_spec = recipe["target_spec"]["grid"]
    assert isinstance(grid_spec, Mapping)
    record = _fixture_record((materialized["train"][0], materialized["validation"][0]))
    atomic_write_json(root / RECIPE_FILENAME, recipe)
    write_acquisition_record(root, **record)
    train_pose, train_range, train_azimuth, _train_elevation = _expected_target_geometry(
        record, materialized["train"][0], grid_spec
    )
    render_grid = target_grid_from_arrays(
        range_m=train_range, azimuth_rad=train_azimuth, expected_grid=grid_spec
    )
    reference = RadarSplatModel(
        torch.tensor([[0.0, 0.0, 0.0], [0.03, 0.01, 0.0]], dtype=torch.float32),
        initial_scale=0.08,
        initial_opacity=0.9,
        initial_noise_probability=0.1,
        initial_reflectance=torch.full((2, 3), 0.99),
        sh_degree=1,
        seed=41,
    )
    train_power = reference.render(
        torch.as_tensor(train_pose, dtype=torch.float32),
        render_grid,
        RadarSplatEffects.b787_clean(),
        active_sh_degree=1,
        gaussian_chunk_size=2,
    )["final_power"][0].detach().cpu().numpy()
    if not np.isfinite(train_power).all() or float(np.max(train_power)) <= 0.0:
        raise AssertionError("synthetic native cache target has no positive renderer power")
    targets = {
        "train": _write_fixture_target(
            root, view_index=materialized["train"][0], role="train", record=record,
            grid_spec=grid_spec, power=train_power,
        ),
        "validation": _write_fixture_target(
            root, view_index=materialized["validation"][0], role="validation", record=record,
            grid_spec=grid_spec, power=train_power * 4.0,
        ),
    }
    atomic_write_json(
        root / MANIFEST_FILENAME,
        {"schema": CACHE_SCHEMA, "version": 1, "roles": materialized},
    )
    atomic_write_json(
        root / STATS_FILENAME,
        {
            "schema": CACHE_SCHEMA,
            "version": 1,
            "fit_split": "train",
            "normalization": "linear_peak",
            "clip": False,
            "train_peak_power": float(np.max(train_power)),
            "occupancy_threshold": OCCUPANCY_THRESHOLD,
        },
    )
    return recipe, record, targets


def _expect_cache_rejection(root: Path, message: str, gates: Gates) -> None:
    try:
        load_cache(root)
    except ValueError:
        gates.check(True, message)
    else:
        raise AssertionError(message)


def _test_cache_and_actual_main(gates: Gates, root: Path) -> None:
    recipe, record, targets = _make_synthetic_cache(root)
    loaded = load_cache(root)
    gates.check(
        loaded.train_indices and loaded.validation_indices
        and float(load_native_view(loaded, loaded.validation_indices[0], "validation", "cpu").target_power.max()) > 1.0,
        "synthetic sealed cache enforces train-only peak while retaining unclipped validation power above one",
    )
    class NonfiniteValidationRenderer:
        def eval(self):
            return self

        def render(self, _pose, render_grid, _effects, **_kwargs):
            return {
                "final_power": torch.full(
                    (1, render_grid.output_azimuth_bins, render_grid.num_range_bins),
                    float("nan"),
                    dtype=torch.float32,
                )
            }

    validation_args = trainer.parse_args(
        ["--cache-root", str(root), "--checkpoint-dir", str(root / "nonfinite_validation")]
    )
    try:
        trainer.validate(
            NonfiniteValidationRenderer(),
            loaded,
            RadarSplatEffects.b787_clean(),
            validation_args,
            torch.device("cpu"),
            active_sh_degree=0,
        )
    except RuntimeError:
        gates.check(True, "validation rejects a non-finite renderer metric before checkpoint serialization")
    else:
        raise AssertionError("validation must reject non-finite renderer output")
    grid = recipe["target_spec"]["grid"]
    assert isinstance(grid, Mapping)
    train_index = loaded.train_indices[0]
    original = targets["train"]
    bad_pose = dict(original)
    bad_pose["sensor_to_world"] = original["sensor_to_world"].copy()
    bad_pose["sensor_to_world"][0, 3] += 0.1
    atomic_save_npz(target_path(root, train_index), **bad_pose)
    _expect_cache_rejection(root, "cache rejects a stored pose that disagrees with direct calibration", gates)
    bad_range = dict(original)
    bad_range["range_m"] = original["range_m"].copy()
    bad_range["range_m"][4] += 1.0e-3
    atomic_save_npz(target_path(root, train_index), **bad_range)
    _expect_cache_rejection(root, "cache rejects a stored range axis that disagrees with direct calibration", gates)
    bad_azimuth = dict(original)
    bad_azimuth["azimuth_rad"] = original["azimuth_rad"].copy()
    bad_azimuth["azimuth_rad"][4] += 1.0e-3
    atomic_save_npz(target_path(root, train_index), **bad_azimuth)
    _expect_cache_rejection(root, "cache rejects a stored azimuth axis that disagrees with direct calibration", gates)
    atomic_save_npz(target_path(root, train_index), **original)
    changed_record = {name: np.asarray(value).copy() for name, value in record.items()}
    # Alter a transverse component, rather than merely lengthening the Tx
    # span along its existing tangent.  The direct calibration normalizes that
    # tangent, so a longitudinal perturbation legitimately leaves the pose
    # unchanged and cannot test the cache rejection path.
    changed_record["tx_pos"][0, -1, 2] += 0.02
    changed_pose, _changed_range, _changed_azimuth, _changed_elevation = _expected_target_geometry(
        changed_record, train_index, grid
    )
    gates.check(
        not np.allclose(
            changed_pose[:3, :3],
            original["sensor_to_world"][:3, :3],
            rtol=0.0,
            atol=1.0e-8,
        ),
        "transverse Tx perturbation changes direct sensor-frame orientation",
    )
    atomic_save_npz(root / ACQUISITION_FILENAME, **changed_record)
    _expect_cache_rejection(root, "cache rejects changed direct Tx calibration even with structurally valid targets", gates)
    changed_center_record = {name: np.asarray(value).copy() for name, value in record.items()}
    changed_center_record["scene_center_m"][0] += 0.02
    atomic_save_npz(root / ACQUISITION_FILENAME, **changed_center_record)
    _expect_cache_rejection(root, "cache rejects a changed direct scene-support centre", gates)
    atomic_save_npz(root / ACQUISITION_FILENAME, **record)
    gates.check(load_cache(root).train_peak_power > 0.0, "restored synthetic cache passes direct calibration audit")

    clean_stop_dir = root / "clean_stop_run"
    clean_stop_argv = [
        "--cache-root", str(root), "--checkpoint-dir", str(clean_stop_dir), "--device", "cpu",
        "--allow-development-subset", "--steps", "2", "--validation-every", "1",
        "--checkpoint-every", "1", "--log-every", "1", "--init-num-gaussians", "2",
        "--init-scale-m", "0.08", "--max-scale-m", "0.2", "--prune-every", "1",
    ]
    original_render = RadarSplatModel.render
    original_emit_phase_resource = trainer._emit_phase_resource
    signal_once = True
    clean_stop_resource_events: list[dict[str, object]] = []

    def render_then_request_stop(self, *args, **kwargs):
        nonlocal signal_once
        rendered = original_render(self, *args, **kwargs)
        if signal_once:
            signal_once = False
            os.kill(os.getpid(), signal.SIGTERM)
        return rendered

    def capture_clean_stop_resource(**kwargs):
        clean_stop_resource_events.append(dict(kwargs))
        return original_emit_phase_resource(**kwargs)

    RadarSplatModel.render = render_then_request_stop
    trainer._emit_phase_resource = capture_clean_stop_resource
    try:
        try:
            trainer.main(clean_stop_argv)
        except SystemExit as exc:
            if exc.code != 143:
                raise
        else:
            raise AssertionError("native trainer must return its clean-stop status after a delivered SIGTERM")
    finally:
        RadarSplatModel.render = original_render
        trainer._emit_phase_resource = original_emit_phase_resource
    clean_stop_checkpoint = trainer._load_checkpoint(clean_stop_dir / "checkpoint_latest.pt", torch.device("cpu"))
    gates.check(
        clean_stop_checkpoint.get("step") == 1
        and clean_stop_checkpoint.get("complete") is False
        and clean_stop_checkpoint.get("finalization_pending") is False
        and not (clean_stop_dir / "checkpoint_final.pt").exists(),
        "delivered SIGTERM saves a resumable native latest checkpoint before terminal finalization",
    )
    gates.check(
        len(clean_stop_resource_events) == 1
        and clean_stop_resource_events[0].get("phase") == "clean_interruption"
        and clean_stop_resource_events[0].get("optimizer_updates_this_invocation") == 1,
        "delivered SIGTERM preserves exact completed-update resource evidence before a resumable exit",
    )
    trainer.main(clean_stop_argv)
    gates.check(
        (clean_stop_dir / "checkpoint_final.pt").is_file()
        and trainer._load_checkpoint(clean_stop_dir / "checkpoint_latest.pt", torch.device("cpu")).get("complete") is True,
        "clean-stop checkpoint resumes through the native trainer to a completed final state",
    )

    run_dir = root / "run"
    run_argv = [
        "--cache-root", str(root), "--checkpoint-dir", str(run_dir), "--device", "cpu",
        "--allow-development-subset", "--steps", "1", "--validation-every", "1",
        "--checkpoint-every", "1", "--log-every", "1", "--init-num-gaussians", "2",
        "--init-scale-m", "0.08", "--max-scale-m", "0.2", "--prune-every", "1",
    ]
    trainer.main(run_argv)
    gates.check(
        (run_dir / "checkpoint_final.pt").is_file()
        and (run_dir / "gaussian_occupancy_geometry.npz").is_file()
        and (run_dir / "summary.json").is_file(),
        "bounded actual-main cache run creates final checkpoint, native geometry, and summary",
    )
    latest_before_rejection = copy.deepcopy(trainer._load_checkpoint(run_dir / "checkpoint_latest.pt", torch.device("cpu")))
    final_before_rejection = copy.deepcopy(trainer._load_checkpoint(run_dir / "checkpoint_final.pt", torch.device("cpu")))
    with np.load(run_dir / "gaussian_occupancy_geometry.npz", allow_pickle=False) as archive:
        geometry_before_rejection = {name: np.asarray(archive[name]).copy() for name in archive.files}
    with (run_dir / "summary.json").open("r", encoding="utf-8") as handle:
        summary_before_rejection = json.load(handle)
    changed_frequency_record = {name: np.asarray(value).copy() for name, value in record.items()}
    changed_frequency_record["frequency_hz"][17] += 1.0e3
    atomic_save_npz(root / ACQUISITION_FILENAME, **changed_frequency_record)
    try:
        trainer.main(run_argv)
    except ValueError:
        gates.check(True, "checkpoint resume rejects changed direct frequency calibration before an update")
    else:
        raise AssertionError("checkpoint resume must reject changed direct frequency calibration")
    _assert_state_equal(
        latest_before_rejection,
        trainer._load_checkpoint(run_dir / "checkpoint_latest.pt", torch.device("cpu")),
        "frequency-rejected latest checkpoint",
    )
    atomic_save_npz(root / ACQUISITION_FILENAME, **record)
    stale_final = copy.deepcopy(final_before_rejection)
    stale_final["step"] = 0
    trainer._atomic_torch_save(run_dir / "checkpoint_final.pt", stale_final)
    try:
        trainer.main(run_argv)
    except ValueError:
        gates.check(True, "stale final checkpoint is rejected before an optimizer update")
    else:
        raise AssertionError("stale final checkpoint must not enter the training loop")
    _assert_state_equal(
        latest_before_rejection,
        trainer._load_checkpoint(run_dir / "checkpoint_latest.pt", torch.device("cpu")),
        "stale-final-rejected latest checkpoint",
    )
    trainer._atomic_torch_save(run_dir / "checkpoint_final.pt", final_before_rejection)
    stale_geometry = {name: value.copy() for name, value in geometry_before_rejection.items()}
    stale_geometry["means"][0, 0] += np.float32(0.05)
    atomic_save_npz(run_dir / "gaussian_occupancy_geometry.npz", **stale_geometry)
    try:
        trainer.main(run_argv)
    except ValueError:
        gates.check(True, "stale existing geometry is rejected without an optimizer update")
    else:
        raise AssertionError("stale geometry must not be silently retained")
    _assert_state_equal(
        latest_before_rejection,
        trainer._load_checkpoint(run_dir / "checkpoint_latest.pt", torch.device("cpu")),
        "stale-geometry-rejected latest checkpoint",
    )
    atomic_save_npz(run_dir / "gaussian_occupancy_geometry.npz", **geometry_before_rejection)
    wrong_dtype_geometry = {name: value.copy() for name, value in geometry_before_rejection.items()}
    wrong_dtype_geometry["means"] = wrong_dtype_geometry["means"].astype(np.float64)
    atomic_save_npz(run_dir / "gaussian_occupancy_geometry.npz", **wrong_dtype_geometry)
    try:
        trainer.main(run_argv)
    except ValueError:
        gates.check(True, "existing geometry with a numerically equal but wrong dtype is rejected")
    else:
        raise AssertionError("geometry dtype changes must not be silently retained")
    atomic_save_npz(run_dir / "gaussian_occupancy_geometry.npz", **geometry_before_rejection)
    stale_summary = dict(summary_before_rejection)
    stale_summary["geometry"] = "not the native Gaussian occupancy readout"
    atomic_write_json(run_dir / "summary.json", stale_summary)
    try:
        trainer.main(run_argv)
    except ValueError:
        gates.check(True, "stale existing summary is rejected without an optimizer update")
    else:
        raise AssertionError("stale summary must not be silently retained")
    _assert_state_equal(
        latest_before_rejection,
        trainer._load_checkpoint(run_dir / "checkpoint_latest.pt", torch.device("cpu")),
        "stale-summary-rejected latest checkpoint",
    )
    atomic_write_json(run_dir / "summary.json", summary_before_rejection)
    pending_latest = copy.deepcopy(final_before_rejection)
    pending_latest["complete"] = False
    pending_latest["finalization_pending"] = True
    trainer._atomic_torch_save(run_dir / "checkpoint_latest.pt", pending_latest)
    trainer.main(run_argv)
    completed_after_pending = trainer._load_checkpoint(run_dir / "checkpoint_latest.pt", torch.device("cpu"))
    gates.check(
        completed_after_pending.get("complete") is True
        and completed_after_pending.get("finalization_pending") is False
        and trainer._directly_equal(completed_after_pending, final_before_rejection),
        "matching final plus finalization-pending latest recovers without another train step",
    )
    nonterminal_latest = copy.deepcopy(final_before_rejection)
    nonterminal_latest["complete"] = False
    nonterminal_latest["finalization_pending"] = False
    trainer._atomic_torch_save(run_dir / "checkpoint_latest.pt", nonterminal_latest)
    try:
        trainer.main(run_argv)
    except ValueError:
        gates.check(True, "final paired with a nonterminal latest is rejected before training")
    else:
        raise AssertionError("nonterminal latest cannot coexist with a final checkpoint")
    _assert_state_equal(
        nonterminal_latest,
        trainer._load_checkpoint(run_dir / "checkpoint_latest.pt", torch.device("cpu")),
        "nonterminal-pair-rejected latest checkpoint",
    )
    trainer._atomic_torch_save(run_dir / "checkpoint_latest.pt", final_before_rejection)
    try:
        trainer.main(run_argv + ["--no-resume"])
    except FileExistsError:
        gates.check(True, "no-resume rejects occupied output before overwriting artifacts")
    else:
        raise AssertionError("no-resume must reject an occupied actual-main fixture")
    (run_dir / "gaussian_occupancy_geometry.npz").unlink()
    (run_dir / "summary.json").unlink()
    trainer.main(run_argv)
    gates.check(
        (run_dir / "gaussian_occupancy_geometry.npz").is_file()
        and (run_dir / "summary.json").is_file(),
        "completed resume recreates only missing derived artifacts without another train step",
    )


def main() -> None:
    gates = Gates()
    _test_direct_comparator(gates)
    _test_target_projection_and_grid(gates)
    _test_sensor_transform(gates)
    _test_normalization_and_balance(gates)
    _test_native_ssim(gates)
    _test_renderer_and_learnability(gates)
    with tempfile.TemporaryDirectory(prefix=".radarsplat_b7873200_native_") as temporary:
        root = Path(temporary)
        _test_prune_and_export(gates, root)
        _test_cache_and_actual_main(gates, root / "synthetic_cache")
    _test_checkpoint_restore(gates)
    print(f"RadarSplat B7873200 native contract passed: {gates.count} checks.", flush=True)


if __name__ == "__main__":
    main()
