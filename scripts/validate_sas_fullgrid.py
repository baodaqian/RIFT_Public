#!/usr/bin/env python
"""Bounded local checks for the native rectangular full-grid sonar smoke."""

from __future__ import annotations

import copy
import json
import math
import os
import sys
import time
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
from rift.rift_sas import ComplexSHSonarField, RIFTSASRectangularGrid
from scripts import diagnose_sas_fullgrid as fullgrid


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


def _node_points(shape: tuple[int, int, int], extent: float = 1.0) -> torch.Tensor:
    axes = [torch.linspace(-extent, extent, size) for size in shape]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)


def _fill_affine_field(field: RIFTSASRectangularGrid) -> None:
    x, y, z = torch.meshgrid(
        *[torch.linspace(-1.0, 1.0, size) for size in field.grid_shape], indexing="ij"
    )
    with torch.no_grad():
        for basis in range(field.w_re.shape[-1]):
            field.w_re[..., basis] = 0.7 + 0.11 * basis + 0.2 * x - 0.13 * y + 0.07 * z
            field.w_im[..., basis] = -0.2 + 0.03 * basis - 0.05 * x + 0.09 * y + 0.04 * z


def test_rectangular_field_contract() -> None:
    shape = (3, 4, 2)
    field = RIFTSASRectangularGrid(shape, 1.0, "cpu", max_degree=3, init_scale=0.0)
    _fill_affine_field(field)
    points = _node_points(shape)
    expected = torch.complex(field.w_re.detach(), field.w_im.detach()).reshape(-1, 16)
    queried = field.query_coefficients(points, chunk_size=2)
    node_error = float((queried - expected).abs().max())
    check(node_error <= 5.0e-7, f"float32 rectangular node accuracy (max error={node_error:.3e})")

    off_node = torch.tensor([[-0.4, 0.2, 0.3]], dtype=torch.float32)
    affine_real = 0.7 + 0.11 * torch.arange(16) + 0.2 * off_node[:, 0] - 0.13 * off_node[:, 1] + 0.07 * off_node[:, 2]
    affine_imag = -0.2 + 0.03 * torch.arange(16) - 0.05 * off_node[:, 0] + 0.09 * off_node[:, 1] + 0.04 * off_node[:, 2]
    expected_off = torch.complex(affine_real, affine_imag)
    check(
        torch.allclose(field.query_coefficients(off_node), expected_off, atol=5.0e-7, rtol=0.0),
        "off-node interpolation is affine and complex-valued",
    )
    near_node = torch.tensor([[5.0e-7, 0.2, 0.3]], dtype=torch.float32, requires_grad=True)
    near_real = 0.7 + 0.11 * torch.arange(16) + 0.2 * near_node[:, 0] - 0.13 * near_node[:, 1] + 0.07 * near_node[:, 2]
    near_imag = -0.2 + 0.03 * torch.arange(16) - 0.05 * near_node[:, 0] + 0.09 * near_node[:, 1] + 0.04 * near_node[:, 2]
    near_expected = torch.complex(near_real, near_imag)
    near_queried = field.query_coefficients(near_node)
    near_gradient = torch.autograd.grad(near_queried.real.sum(), near_node)[0]
    check(
        torch.allclose(near_queried, near_expected, atol=5.0e-7, rtol=0.0),
        "near-lattice off-node interpolation remains affine",
    )
    check(abs(float(near_gradient[0, 0])) > 0.1, "near-lattice interpolation retains coordinate derivative")
    outside = field.query_coefficients(torch.tensor([[1.01, 0.0, 0.0], [-1.1, 0.0, 0.0]]))
    check(bool(torch.equal(outside, torch.zeros_like(outside))), "outside-box queries are zero")
    check(
        torch.allclose(field.query_coefficients(points, chunk_size=0), field.query_coefficients(points, chunk_size=3)),
        "positive query chunks match the unchunked field query",
    )
    check(
        torch.allclose(field.query_dc(off_node), torch.complex(affine_real[:1], affine_imag[:1])),
        "complex DC is interpolated before any magnitude operation",
    )
    adapter = ComplexSHSonarField(field, torch.tensor([-2.0] * 3), torch.tensor([2.0] * 3), 3, query_chunk=2)
    physical = off_node * 2.0
    expected_density = torch.complex(affine_real[:1], affine_imag[:1]).abs() * (1.0 / math.sqrt(4.0 * math.pi))
    check(
        torch.allclose(adapter.query_density(physical), expected_density, atol=5.0e-7, rtol=0.0),
        "physical adapter applies DC magnitude and Y00 after complex interpolation",
    )

    field.zero_grad(set_to_none=True)
    loss = field.query_coefficients(points, chunk_size=2).real.sum() + field.query_coefficients(points, chunk_size=2).imag.sum()
    loss.backward()
    check(bool((field.w_re.grad.abs().sum(dim=-1) > 0).all()), "every site receives real-coefficient gradient")
    check(bool((field.w_im.grad.abs().sum(dim=-1) > 0).all()), "every site receives imaginary-coefficient gradient")


def _make_sonar_model(shape=(5, 4, 3)):
    field = RIFTSASRectangularGrid(shape, 1.0, "cpu", max_degree=3, init_scale=0.0)
    x, y, z = torch.meshgrid(
        *[torch.linspace(-1.0, 1.0, size) for size in shape], indexing="ij"
    )
    with torch.no_grad():
        field.w_re[..., 0] = 0.8 + 0.2 * x + 0.11 * y + 0.05 * z
        field.w_im[..., 0] = 0.12 + 0.04 * x - 0.03 * y + 0.02 * z
        for basis in range(1, 16):
            field.w_re[..., basis] = 0.02 * (basis + 1) + 0.01 * x
            field.w_im[..., basis] = -0.01 * (basis + 1) + 0.01 * z
    return ComplexSHSonarField(field, torch.tensor([-1.0] * 3), torch.tensor([1.0] * 3), 3, query_chunk=32)


def _make_calibration() -> torch.nn.Module:
    calibration = train_sas.LogPolarCalibration()
    with torch.no_grad():
        calibration.log_mag.fill_(0.17)
        calibration.phase.fill_(0.23)
        calibration.initialized.fill_(True)
    return calibration


def _render_case(selected: torch.Tensor, ray_chunk: int):
    model = _make_sonar_model()
    calibration = _make_calibration()
    radii = torch.linspace(2.4, 3.0, 7)
    tx = torch.tensor([2.0, 0.1, 0.2])
    rx = torch.tensor([2.1, -0.15, 0.25])
    corners = torch.tensor([
        [-0.8, -0.8, -0.8], [-0.8, -0.8, 0.8], [-0.8, 0.8, -0.8], [-0.8, 0.8, 0.8],
        [0.8, -0.8, -0.8], [0.8, -0.8, 0.8], [0.8, 0.8, -0.8], [0.8, 0.8, 0.8],
    ])
    prediction, aux = sas_operator.render_sas_bins(
        model,
        radii,
        tx,
        rx,
        corners,
        num_rays=81,
        opacity_scale=500.0,
        lambertian_ratio=0.25,
        normal_step=0.0032,
        beamwidth_deg=30.0,
        output_bin_indices=selected,
        sh_direction="rx_to_point",
        ray_chunk=ray_chunk,
    )
    target = torch.linspace(0.4, 1.0, selected.numel(), dtype=torch.float32).to(torch.complex64)
    calibrated = calibration(prediction)
    loss = (calibrated.real - target.real).square().mean() + (calibrated.imag - target.imag).square().mean()
    loss.backward()
    grads = (
        model.coefficient_field.w_re.grad.detach().clone(),
        model.coefficient_field.w_im.grad.detach().clone(),
        calibration.log_mag.grad.detach().clone(),
        calibration.phase.grad.detach().clone(),
    )
    return prediction.detach(), grads, aux


def test_ray_chunk_equivalence() -> None:
    selections = {
        "all": torch.arange(7, dtype=torch.long),
        "sorted_subset": torch.tensor([0, 2, 5], dtype=torch.long),
        "full_permutation": torch.tensor([6, 0, 3, 1, 5, 2, 4], dtype=torch.long),
    }
    max_prediction_error = 0.0
    max_gradient_error = 0.0
    for name, selected in selections.items():
        baseline, base_grads, base_aux = _render_case(selected, 0)
        for ray_chunk in (1, 7, 1000):
            prediction, grads, aux = _render_case(selected, ray_chunk)
            prediction_error = float((prediction - baseline).abs().max())
            gradient_error = max(
                float((grads[index] - base_grads[index]).abs().max())
                for index in range(4)
            )
            max_prediction_error = max(max_prediction_error, prediction_error)
            max_gradient_error = max(max_gradient_error, gradient_error)
            check(
                torch.allclose(prediction, baseline, atol=2.0e-5, rtol=3.0e-4),
                f"{name} ray_chunk={ray_chunk} prediction matches unchunked",
            )
            check(
                all(torch.allclose(grads[index], base_grads[index], atol=2.0e-5, rtol=3.0e-4) for index in range(4)),
                f"{name} ray_chunk={ray_chunk} coefficient/gain gradients match",
            )
            check(
                not aux["transmittance"].requires_grad and not aux["lambertian"].requires_grad,
                f"{name} ray_chunk={ray_chunk} auxiliary fields are detached",
            )
    check(max_prediction_error < 2.0e-5, f"ray-chunk maximum prediction error recorded ({max_prediction_error:.3e})")
    check(max_gradient_error < 2.0e-4, f"ray-chunk maximum gradient error recorded ({max_gradient_error:.3e})")
    expect_raises(
        ValueError,
        lambda: _render_case(torch.arange(7), -1),
        "negative ray chunk is rejected",
    )
    expect_raises(
        ValueError,
        lambda: sas_operator.render_sas_bins(
            _make_sonar_model(), torch.linspace(2.4, 3.0, 7), torch.tensor([2.0, 0.1, 0.2]),
            torch.tensor([2.1, -0.15, 0.25]), torch.tensor([[-0.8] * 3, [0.8] * 3]),
            num_rays=4, opacity_scale=1.0, mean_normalize_opacity=True, ray_chunk=1,
        ),
        "ray chunk rejects globally normalized opacity",
    )

    counted = [0]
    original = sas_operator._render_sas_ray_chunk

    def counted_chunk(*args, **kwargs):
        counted[0] += 1
        return original(*args, **kwargs)

    model = _make_sonar_model()
    calibration = _make_calibration()
    radii = torch.linspace(2.4, 3.0, 7)
    tx = torch.tensor([2.0, 0.1, 0.2])
    rx = torch.tensor([2.1, -0.15, 0.25])
    corners = torch.tensor([
        [-0.8, -0.8, -0.8], [-0.8, -0.8, 0.8], [-0.8, 0.8, -0.8], [-0.8, 0.8, 0.8],
        [0.8, -0.8, -0.8], [0.8, -0.8, 0.8], [0.8, 0.8, -0.8], [0.8, 0.8, 0.8],
    ])
    with mock.patch.object(sas_operator, "_render_sas_ray_chunk", side_effect=counted_chunk):
        prediction, aux = sas_operator.render_sas_bins(
            model,
            radii,
            tx,
            rx,
            corners,
            num_rays=81,
            opacity_scale=500.0,
            lambertian_ratio=0.25,
            normal_step=0.0032,
            beamwidth_deg=30.0,
            output_bin_indices=torch.arange(7),
            sh_direction="rx_to_point",
            ray_chunk=7,
        )
        forward_calls = counted[0]
        expected_chunks = math.ceil(int(aux["actual_rays"].item()) / 7)
        check(
            forward_calls == expected_chunks,
            f"ray checkpoint forward calls equal chunk count ({forward_calls}={expected_chunks})",
        )
        target = torch.linspace(0.4, 1.0, 7, dtype=torch.float32).to(torch.complex64)
        calibrated = calibration(prediction)
        loss = (calibrated.real - target.real).square().mean() + (calibrated.imag - target.imag).square().mean()
        loss.backward()
    check(counted[0] > forward_calls, f"non-reentrant ray checkpoint recomputes pure chunks during backward (forward={forward_calls}, total={counted[0]})")


def _optimizer_update_pair(ray_chunk_a: int, ray_chunk_b: int):
    base = _make_sonar_model()
    base_state = copy.deepcopy(base.state_dict())
    base_calibration = _make_calibration()
    calibration_state = copy.deepcopy(base_calibration.state_dict())
    cache = SimpleNamespace(
        radii=np.linspace(2.4, 3.0, 7, dtype=np.float32),
        tx_coords=np.asarray([[2.0, 0.1, 0.2]], dtype=np.float32),
        rx_coords=np.asarray([[2.1, -0.15, 0.25]], dtype=np.float32),
        corners=np.asarray([
            [-0.8, -0.8, -0.8], [-0.8, -0.8, 0.8], [-0.8, 0.8, -0.8], [-0.8, 0.8, 0.8],
            [0.8, -0.8, -0.8], [0.8, -0.8, 0.8], [0.8, 0.8, -0.8], [0.8, 0.8, 0.8],
        ], dtype=np.float32),
        weights=np.full((1, 7), 0.3 + 0.2j, dtype=np.complex64),
        tx_vecs=None,
    )
    common = SimpleNamespace(
        num_rays=81, opacity_scale=500.0, lambertian_ratio=0.25, normal_step=0.0032,
        beamwidth_deg=30.0, opacity_normalize=False, sh_direction="rx_to_point",
        signal_scale=1.0, query_chunk=32,
    )
    models = []
    calibrations = []
    optimizers = []
    for ray_chunk in (ray_chunk_a, ray_chunk_b):
        model = _make_sonar_model()
        model.load_state_dict(base_state)
        calibration = _make_calibration()
        calibration.load_state_dict(calibration_state)
        optimizer = torch.optim.Adam(list(model.parameters()) + list(calibration.parameters()), lr=1.0e-3)
        models.append(model)
        calibrations.append(calibration)
        optimizers.append(optimizer)
        common_args = copy.copy(common)
        common_args.ray_chunk = ray_chunk
        for _step in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss, _metrics, _aux = train_sas.render_one(
                model, calibration, cache, 0, np.arange(7, dtype=np.int64), common_args,
                torch.device("cpu"), allow_calibration_init=False,
            )
            loss.backward()
            optimizer.step()
    max_parameter_error = max(
        float((left - right).abs().max())
        for left, right in zip(models[0].parameters(), models[1].parameters())
    )
    check(
        max_parameter_error <= 2.0e-4,
        f"two optimizer updates match across ray decomposition (max={max_parameter_error:.3e})",
    )
    check(torch.allclose(calibrations[0].log_mag, calibrations[1].log_mag, atol=2.0e-4, rtol=2.0e-3), "optimizer gain log-magnitude matches")
    check(torch.allclose(calibrations[0].phase, calibrations[1].phase, atol=2.0e-4, rtol=2.0e-3), "optimizer gain phase matches")


def _reference_checkpoint(path: Path) -> None:
    args = train_sas.parse_args([
        "--cache", "saved-cache", "--model", "adaptive_rift_sas", "--checkpoint-name", "saved",
        "--device", "cpu", "--steps", "1100", "--lr", "0.001", "--coefficient-lr", "0.001",
        "--granularity", "64", "--initial-granularity", "16", "--adaptive-capacity", "65536",
        "--max-active", "65536", "--sh-degree", "3", "--refine-every", "1000",
        "--num-rays", "4900", "--max-bins", "110", "--opacity-scale", "500",
        "--normal-step", "0.0032", "--signal-scale", "10", "--lambertian-ratio", "0",
        "--beamwidth-deg", "30", "--grad-clip", "1", "--eval-every", "100",
        "--eval-pings", "8", "--eval-bins", "0", "--max-pings", "0", "--seed", "42",
        "--require-explicit-splits", "--calibration-mode", "log_polar", "--no-opacity-normalize",
    ])
    torch.save({
        "step": 700,
        "model_kind": "adaptive_rift_sas",
        "calibration_mode": "log_polar",
        "args": vars(args),
        "model_state_dict": {"reference_sentinel": torch.tensor([123.0])},
    }, path)


def test_driver_recipe_and_source_only_reference(tmp: Path) -> None:
    reference = tmp / "reference.pt"
    _reference_checkpoint(reference)
    state = torch.load(reference, map_location="cpu", weights_only=False)
    saved_args = fullgrid._validate_reference_state(state)
    recipe = fullgrid._build_recipe(
        saved_args, cache=Path("cache"), reference=reference,
        output=tmp / "output", device="cpu",
    )
    parsed = train_sas.parse_args(recipe)
    check("--resume" not in recipe, "full-grid recipe removes reference resume pair")
    check(parsed.model == "rift_sas", "full-grid recipe selects fixed-grid RIFT-SAS")
    check(tuple(parsed.grid_shape) == fullgrid.GRID_SHAPE, "full-grid recipe selects native rectangular shape")
    check(parsed.ray_chunk == 128 and parsed.query_chunk == 65536, "full-grid recipe selects bounded runtime chunks")
    check(parsed.steps == 900 and parsed.eval_every == 100 and parsed.eval_pings == 8 and parsed.eval_bins == 0, "full-grid recipe preserves matched readouts")
    check(parsed.lr == 0.001 and parsed.calibration_mode == "log_polar", "full-grid recipe uses saved coefficient scale and log-polar gain")
    check(state["model_state_dict"]["reference_sentinel"].item() == 123.0, "reference checkpoint fixture contains a sentinel tensor")

    incompatible = copy.deepcopy(state)
    incompatible["args"]["num_rays"] = 4899
    expect_raises(ValueError, lambda: fullgrid._validate_reference_state(incompatible), "incompatible reference physics is rejected")
    existing = tmp / "existing"
    existing.mkdir()
    expect_raises(
        FileExistsError,
        lambda: fullgrid.main([
            "--cache", "cache", "--reference-checkpoint", str(reference), "--output", str(existing), "--device", "cpu"
        ]),
        "existing full-grid output is rejected",
    )
    captured = {}

    def fake_main(argv, *, diagnostic_observer):
        captured["argv"] = list(argv)
        diagnostic_observer._finished = True

    with mock.patch.object(fullgrid.train_sas, "main", side_effect=fake_main):
        fullgrid.main([
            "--cache", "cache", "--reference-checkpoint", str(reference),
            "--output", str(tmp / "source_only"), "--device", "cpu",
        ])
    check("--resume" not in captured["argv"], "driver passes no reference tensors or resume path to training")


def _synthetic_cache() -> SimpleNamespace:
    return SimpleNamespace(
        manifest={"geometry_grid_shape": [3, 4, 2], "dataset_identity": "synthetic_fullgrid"},
        has_explicit_splits=True,
        num_pings=4,
        num_bins=7,
        weights=np.full((4, 7), 0.3 + 0.2j, dtype=np.complex64),
        tx_coords=np.asarray([[2.0, 0.1, 0.2]] * 4, dtype=np.float32),
        rx_coords=np.asarray([[2.1, -0.15, 0.25]] * 4, dtype=np.float32),
        tx_vecs=None,
        radii=np.linspace(2.4, 3.0, 7, dtype=np.float32),
        corners=np.asarray([
            [-1.0, -1.0, -1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [-1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0], [1.0, -1.0, 1.0], [1.0, 1.0, -1.0], [1.0, 1.0, 1.0],
        ], dtype=np.float32),
        voxels=np.zeros((1, 3), dtype=np.float32),
        source_ids=np.asarray([500, 501, 99, 100], dtype=np.int64),
        train_indices=np.asarray([0, 1], dtype=np.int64),
        validation_indices=np.asarray([2], dtype=np.int64),
        test_indices=np.asarray([3], dtype=np.int64),
    )


def test_small_actual_main_footer_and_checkpoint(tmp: Path) -> None:
    cache = _synthetic_cache()
    observer = fullgrid.FullGridDiagnosticObserver(
        tmp / "small_output", "fixture-reference", expected_grid_shape=(3, 4, 2),
        expected_source_ids=np.asarray([99]), expected_train_count=2, expected_bin_count=7,
        expected_end_step=2,
        expected_dataset_identity="synthetic_fullgrid", expected_bounds=((-1.0,) * 3, (1.0,) * 3),
    )
    eval_roles = []

    def fake_render(model, _calibration, _cache, _ping, bins, _args, _device, **_kwargs):
        field = model.coefficient_field
        loss = field.w_re.square().mean() + field.w_im.square().mean()
        predicted = torch.complex(field.w_re[..., 0].mean().expand(len(bins)), field.w_im[..., 0].mean().expand(len(bins)))
        target = torch.full_like(predicted, 0.3 + 0.2j)
        return loss, train_sas.metric_record(predicted, target), {
            "calibration_raw": predicted.detach(), "calibration_target": target.detach(),
            "calibration_predicted": predicted.detach(),
            "transmittance": torch.ones((len(bins), 1)), "lambertian": torch.ones((len(bins), 1)),
            "actual_rays": 1,
        }

    def fake_evaluate(_model, _calibration, _cache, role_indices, _args, _device):
        eval_roles.append(np.asarray(role_indices).tolist())
        return {
            "rel_mse": 0.5, "l1_real": 0.0, "l1_imag": 0.0, "l1_mag": 0.0,
            "mse_real": 0.0, "mse_imag": 0.0, "mse_mag": 0.0,
            "complete": 1.0, "views": 1.0,
        }

    argv = [
        "--cache", "synthetic", "--model", "rift_sas", "--checkpoint-root", str(tmp),
        "--checkpoint-name", "small_output", "--device", "cpu", "--steps", "2",
        "--grid-shape", "3", "4", "2", "--granularity", "4", "--sh-degree", "3",
        "--num-rays", "4", "--max-bins", "0", "--eval-every", "1", "--eval-pings", "8",
        "--eval-bins", "0", "--checkpoint-every", "1", "--log-every", "100",
        "--seed", "3", "--calibration-mode", "log_polar", "--ray-chunk", "1",
        "--query-chunk", "8", "--signal-scale", "1", "--opacity-scale", "10",
        "--normal-step", "0.1", "--no-opacity-normalize",
    ]
    with mock.patch.object(train_sas, "load_sas_cache", return_value=cache), \
         mock.patch.object(train_sas, "render_one", side_effect=fake_render), \
         mock.patch.object(train_sas, "evaluate", side_effect=fake_evaluate):
        train_sas.main(argv, diagnostic_observer=observer)
    check(observer._finished, "small actual trainer reaches the full-grid diagnostic footer")
    check(eval_roles == [[2], [2]], "diagnostic footer prevents TEST evaluation")
    summary = json.loads((tmp / "small_output" / "fullgrid_summary.json").read_text(encoding="utf-8"))
    check(summary["grid_shape"] == [3, 4, 2] and summary["gradient_support_count"] == 24, "summary records full learned support and gradients")
    check((tmp / "small_output" / "checkpoint_final_diagnostic.pt").exists(), "diagnostic checkpoint is serialized")
    checkpoint = torch.load(tmp / "small_output" / "checkpoint_final_diagnostic.pt", map_location="cpu", weights_only=False)
    check(checkpoint["args"]["grid_shape"] == [3, 4, 2], "rectangular shape round-trips through checkpoint args")
    mismatch_args = SimpleNamespace(**copy.deepcopy(checkpoint["args"]))
    mismatch_args.grid_shape = [3, 4, 3]
    expect_raises(
        ValueError,
        lambda: train_sas._reconcile_saved_recipe(
            mismatch_args, checkpoint, {"grid_shape"}, set(), eval_only=False
        ),
        "explicit rectangular shape mismatch is rejected on restore",
    )
    legacy = copy.deepcopy(checkpoint)
    del legacy["args"]["grid_shape"]
    check(train_sas._state_recipe_contract(legacy)["grid_shape"] is None, "legacy cubic checkpoint migrates only missing grid_shape to None")


def main() -> None:
    test_rectangular_field_contract()
    test_ray_chunk_equivalence()
    tmp = PROJECT_ROOT / "tmp" / f"fullgrid_validate_{os.getpid()}_{time.time_ns()}"
    tmp.mkdir(parents=True, exist_ok=False)
    _optimizer_update_pair(0, 1)
    test_driver_recipe_and_source_only_reference(tmp)
    test_small_actual_main_footer_and_checkpoint(tmp)
    print("All full-grid sonar gates passed.")


if __name__ == "__main__":
    main()
