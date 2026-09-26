#!/usr/bin/env python
"""Focused data-free and allocated-node gates for the SpINR-style B787 path.

The default mode creates only tiny temporary synthetic files and never opens a
PACE dataset or writes an experiment artifact.  Passing ``--allocated-pace``
is an explicit request for the real authorized B787 train/validation ingest
and one all-bin/all-pair GPU update; it must be run by an allocated-node
launcher, never on a login node.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train as legacy_train  # noqa: E402
import train_spinr_style as spinr_train  # noqa: E402
import rift.spinr_style as spinr_module  # noqa: E402
from rift.config import cc  # noqa: E402
from rift.forward_operator import get_kvector  # noqa: E402
from rift.range_operator import range_adjoint_operator, range_forward_operator  # noqa: E402
from rift.spinr_style import (  # noqa: E402
    SPINR_STYLE_HIDDEN_LAYERS,
    SPINR_STYLE_HIDDEN_WIDTH,
    SPINR_STYLE_INPUT_FEATURES,
    SPINR_STYLE_PARAMETER_COUNT,
    SPINR_STYLE_PHASE_SIGN,
    SPINR_STYLE_RANGE_MODEL,
    SPINR_STYLE_SUPPORT_M,
    SealedRawComplexViews,
    SpinrStyleINR,
    build_spinr_style_acquisition_identity,
    encode_spinr_style_coordinates,
    frequency_to_range_bins,
    metadata_frequency_grid,
    midpoint_grid,
    npz_tx_rx_frequency_to_renderer,
    range_to_frequency_cotangent,
    scale_field_to_renderer_weights,
    spinr_style_objective,
    validate_spinr_style_acquisition_identity,
)


EPS = 1e-12


class Gate:
    def __init__(self) -> None:
        self.count = 0

    def check(self, condition: bool, message: str) -> None:
        self.count += 1
        if not condition:
            raise AssertionError(message)
        print(f"  PASS: {message}", flush=True)

    def near(self, actual: torch.Tensor | float, expected: torch.Tensor | float, tolerance: float, message: str) -> None:
        actual_tensor = torch.as_tensor(actual)
        expected_tensor = torch.as_tensor(expected, device=actual_tensor.device)
        denominator = expected_tensor.norm().clamp_min(EPS)
        relative = float((actual_tensor - expected_tensor).norm() / denominator)
        self.check(relative <= tolerance, f"{message} (relative L2={relative:.3e}, tolerance={tolerance:.1e})")


def _tree_equal(left: Any, right: Any) -> bool:
    """Exact structural equality for checkpoint primitives plus Torch tensors."""

    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping) or set(left) != set(right):
            return False
        return all(_tree_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            isinstance(left, (list, tuple)) and isinstance(right, (list, tuple))
            and len(left) == len(right)
            and all(_tree_equal(a, b) for a, b in zip(left, right))
        )
    return left == right


def _direct_product_forward(
    frequencies: torch.Tensor,
    kvector: torch.Tensor,
    rx_pos: torch.Tensor,
    tx_pos: torch.Tensor,
    points: torch.Tensor,
    weights: torch.Tensor,
    *,
    phase_sign: float,
) -> torch.Tensor:
    """Independent complex128 product-spreading reference, [F,Rx,Tx]."""

    points = points.to(dtype=torch.float64)
    rx_pos = rx_pos.to(device=points.device, dtype=torch.float64)
    tx_pos = tx_pos.to(device=points.device, dtype=torch.float64)
    kvector = kvector.to(device=points.device, dtype=torch.float64)
    weights = weights.to(device=points.device, dtype=torch.complex128)
    r_tx = torch.linalg.norm(points[:, None, :] - tx_pos[None, :, :], dim=-1).clamp_min(1e-9)
    r_rx = torch.linalg.norm(points[:, None, :] - rx_pos[None, :, :], dim=-1).clamp_min(1e-9)
    r_sum = r_tx[:, :, None] + r_rx[:, None, :]
    geometry = 1.0 / (r_tx[:, :, None] * r_rx[:, None, :])
    g_const = 1.0 / ((4.0 * torch.pi) ** 2)
    output = []
    for kvalue in kvector:
        kernel = g_const * geometry * torch.exp(1j * phase_sign * kvalue * r_sum)
        output.append((weights[:, None, None] * kernel).sum(dim=0).transpose(0, 1))
    return torch.stack(output, dim=0)


def _direct_product_adjoint(
    frequencies: torch.Tensor,
    kvector: torch.Tensor,
    rx_pos: torch.Tensor,
    tx_pos: torch.Tensor,
    points: torch.Tensor,
    residual: torch.Tensor,
    *,
    phase_sign: float,
) -> torch.Tensor:
    del frequencies  # kvector is the exact physics coordinate used below.
    points = points.to(dtype=torch.float64)
    rx_pos = rx_pos.to(device=points.device, dtype=torch.float64)
    tx_pos = tx_pos.to(device=points.device, dtype=torch.float64)
    residual = residual.to(device=points.device, dtype=torch.complex128)
    r_tx = torch.linalg.norm(points[:, None, :] - tx_pos[None, :, :], dim=-1).clamp_min(1e-9)
    r_rx = torch.linalg.norm(points[:, None, :] - rx_pos[None, :, :], dim=-1).clamp_min(1e-9)
    r_sum = r_tx[:, :, None] + r_rx[:, None, :]
    geometry = 1.0 / (r_tx[:, :, None] * r_rx[:, None, :])
    g_const = 1.0 / ((4.0 * torch.pi) ** 2)
    result = torch.zeros(points.shape[0], device=points.device, dtype=torch.complex128)
    for frequency_index, kvalue in enumerate(kvector.to(device=points.device, dtype=torch.float64)):
        kernel = g_const * geometry * torch.exp(-1j * phase_sign * kvalue * r_sum)
        result += torch.einsum("ptr,rt->p", kernel, residual[frequency_index])
    return result


def _tiny_physics(device: torch.device) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cpu").manual_seed(11)
    frequencies = 8.5e9 + torch.arange(16, device=device, dtype=torch.float64) * 5.0e6
    kvector = get_kvector(frequencies, cc).to(dtype=torch.float64)
    # Deliberately asymmetric channel counts and non-symmetric locations make
    # a Tx/Rx permutation bug visible.
    tx_pos = torch.tensor([[1.8, -0.12, 0.04], [1.75, 0.10, -0.06]], device=device, dtype=torch.float64)
    rx_pos = torch.tensor(
        [[1.9, -0.08, 0.02], [1.82, 0.13, 0.08], [1.78, -0.04, -0.10]],
        device=device,
        dtype=torch.float64,
    )
    points = torch.tensor(
        [[-0.05, 0.01, 0.00], [0.03, -0.04, 0.02], [0.01, 0.04, -0.03]],
        device=device,
        dtype=torch.float64,
    )
    weights = (
        torch.randn(points.shape[0], generator=generator, dtype=torch.float64, device=device)
        + 1j * torch.randn(points.shape[0], generator=generator, dtype=torch.float64, device=device)
    )
    return frequencies, kvector, rx_pos, tx_pos, points, weights


def stage_recipe_and_encoding(gate: Gate) -> None:
    print("Stage A: fixed neural recipe and FP64-to-FP32 boundary", flush=True)
    torch.manual_seed(42)
    model = SpinrStyleINR()
    gate.check(model.trainable_parameter_count() == SPINR_STYLE_PARAMETER_COUNT, "exact 3,566,641 trainable parameters")
    gate.check(
        len(model.hidden) == SPINR_STYLE_HIDDEN_LAYERS
        and all(layer.in_features == (SPINR_STYLE_INPUT_FEATURES if index == 0 else SPINR_STYLE_HIDDEN_WIDTH)
                and layer.out_features == SPINR_STYLE_HIDDEN_WIDTH for index, layer in enumerate(model.hidden)),
        "six 840-wide hidden linear layers are present",
    )
    gate.check(model.head.out_features == 1 and model.head.bias is not None, "scalar signed-real linear head has a bias")
    expected_names = {
        *(f"hidden.{index}.weight" for index in range(6)),
        *(f"hidden.{index}.bias" for index in range(6)),
        "head.weight", "head.bias",
    }
    gate.check(set(dict(model.named_parameters())) == expected_names, "no gain, latent, view, frequency, or padding parameters")
    coordinates64, _ = midpoint_grid(3, dtype=torch.float64)
    output = model(coordinates64)
    gate.check(output.dtype == torch.float32 and output.shape == (27,), "FP64 quadrature feeds the FP32 network safely")
    loss = output.square().mean()
    loss.backward()
    gate.check(
        all(parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.norm() > 0
            for parameter in model.parameters()),
        "every trainable tensor receives a finite nonzero gradient",
    )
    encoded_zero = encode_spinr_style_coordinates(torch.zeros(1, 3, dtype=torch.float64))
    gate.check(encoded_zero.shape[-1] == SPINR_STYLE_INPUT_FEATURES, "coordinate encoding has exactly 39 features")
    gate.check(torch.equal(encoded_zero[:, :3], torch.zeros(1, 3, dtype=torch.float64)), "raw normalized zero coordinates remain zero")
    gate.check(
        all(torch.equal(encoded_zero[:, 3 + 6 * band:6 + 6 * band], torch.zeros(1, 3, dtype=torch.float64))
            for band in range(6)),
        "all sine features are zero at origin",
    )
    gate.check(
        all(torch.equal(encoded_zero[:, 6 + 6 * band:9 + 6 * band], torch.ones(1, 3, dtype=torch.float64))
            for band in range(6)),
        "all cosine features are one at origin",
    )
    try:
        encode_spinr_style_coordinates(torch.tensor([[SPINR_STYLE_SUPPORT_M * 1.01, 0.0, 0.0]]))
    except ValueError:
        rejected = True
    else:
        rejected = False
    gate.check(rejected, "out-of-support coordinates are rejected rather than clipped")


def stage_frequency_fft_and_axis(gate: Gate) -> None:
    print("Stage B: metadata frequency grid, transform, and Tx/Rx ordering", flush=True)
    meta = {"radar_fc_hz": 10.0e9, "radar_bandwidth_hz": 3.0e9, "num_adc_samples": 600}
    frequencies = metadata_frequency_grid(meta)
    gate.check(frequencies.dtype == torch.float64 and frequencies.shape == (600,), "metadata frequency grid is float64 with all 600 bins")
    gate.check(float(frequencies[0]) == 8.5e9 and float(frequencies[-1]) == 11.495e9, "frequency grid uses fc-BW/2+nBW/N, not an endpoint linspace")
    raw = torch.empty(2, 3, 5, dtype=torch.complex128)
    for tx in range(2):
        for rx in range(3):
            for frequency in range(5):
                raw[tx, rx, frequency] = complex(100 * tx + 10 * rx + frequency, -frequency)
    rendered = npz_tx_rx_frequency_to_renderer(raw)
    gate.check(
        rendered.shape == (5, 3, 2) and all(rendered[f, r, t] == raw[t, r, f]
                                               for f in range(5) for r in range(3) for t in range(2)),
        "asymmetric 2-Tx x 3-Rx sentinel preserves [Tx,Rx,F] -> [F,Rx,Tx] exactly",
    )
    torch.manual_seed(5)
    signal_tensor = torch.randn(4, 7, 3, 2, dtype=torch.float64) + 1j * torch.randn(4, 7, 3, 2, dtype=torch.float64)
    transformed = frequency_to_range_bins(signal_tensor)
    cotangent = torch.randn_like(transformed)
    left = torch.vdot(transformed.reshape(-1), cotangent.reshape(-1))
    right = torch.vdot(signal_tensor.reshape(-1), range_to_frequency_cotangent(cotangent).reshape(-1))
    gate.near(left, right, 5e-13, "forward-normalized FFT VJP preserves the complex inner product with batch axes")
    power = float(signal_tensor.abs().square().mean().item())
    zero_loss = spinr_style_objective(torch.zeros_like(signal_tensor), signal_tensor, training_mean_raw_power=power)
    gate.check(abs(float(zero_loss) - 1.5) < 5e-12, "all-bin zero predictor has the frozen normalized loss 1.5")
    gate.check(float(spinr_style_objective(signal_tensor, signal_tensor, training_mean_raw_power=power)) == 0.0,
               "perfect prediction has zero native spectral objective")


def _write_tiny_sealed_npz(root: Path) -> tuple[Path, Path]:
    path = root / "tiny_sealed.npz"
    response = np.zeros((5, 2, 3, 1, 4), dtype=np.complex64)
    response[:, :, :, :, :] = 1.0 + 2.0j
    positions = np.zeros((5, 3), dtype=np.float64)
    tx = np.zeros((5, 2, 3), dtype=np.float64)
    rx = np.zeros((5, 3, 3), dtype=np.float64)
    metadata = {"radar_fc_hz": 10.0e9, "radar_bandwidth_hz": 1.0e9, "num_adc_samples": 4}
    np.savez_compressed(
        path,
        response=response,
        viewpoint_positions=positions,
        tx_pos=tx,
        rx_pos=rx,
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    manifest = {
        "schema_version": 1,
        "name": "tiny_sealed",
        "dataset": {"num_views": 5, "response_shape": [5, 2, 3, 1, 4], "response_dtype": "complex64"},
        "split": {
            "complete_partition": True,
            "test_sealed": True,
            "strategy": "synthetic",
            "num_train": 2,
            "num_validation": 1,
            "num_test": 1,
            "num_unused": 1,
            "unused_sealed": True,
            "train_indices": [3, 1],
            "validation_indices": [4],
            "test_indices": [0],
            "unused_indices": [2],
        },
    }
    manifest_path = root / "tiny_sealed.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest_path


def stage_sealed_header_preflight(gate: Gate) -> None:
    print("Stage C: sealed manifest/header preflight is payload-free", flush=True)
    with tempfile.TemporaryDirectory(prefix="rift_spinr_sealed_") as temporary:
        path, manifest = _write_tiny_sealed_npz(Path(temporary))
        calls: list[bool] = []
        original_loader = legacy_train.load_npz_arrays

        def recording_loader(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
            calls.append(bool(kwargs.get("load_response", True)))
            return original_loader(*args, **kwargs)

        legacy_train.load_npz_arrays = recording_loader
        try:
            _arrays, contract = legacy_train._load_sealed_npz_protocol_contract(
                path, manifest, num_train=2, num_val=1, num_test=1)
        finally:
            legacy_train.load_npz_arrays = original_loader
        gate.check(calls == [False], "sealed contract is validated before any response payload materialization")
        gate.check(
            contract["role_ids"]["train"] == [3, 1]
            and contract["response_access"]["reserved_test_materialized"] is False
            and contract["response_access"]["unused_materialized"] is False,
            "sealed contract preserves role order and refuses test/unused access",
        )
        broken = json.loads(manifest.read_text(encoding="utf-8"))
        broken["split"]["test_indices"] = [1]
        manifest.write_text(json.dumps(broken), encoding="utf-8")
        try:
            legacy_train._load_sealed_npz_protocol_contract(path, manifest, num_train=2, num_val=1, num_test=1)
        except ValueError:
            rejected = True
        else:
            rejected = False
        gate.check(rejected, "overlapping sealed roles fail header/manifest validation")


def stage_raw_complex_adapter(gate: Gate) -> None:
    """Exercise the actual raw-complex adapter ingress and access methods.

    A full production adapter intentionally retains 4,200 authorized B787
    views, which is unsuitable for a default data-free test.  This bounded
    fixture uses the same one-view ingress helper and the real adapter methods
    to test dtype/axis/precision/sealed-role semantics without allocating a
    multi-GiB cache.
    """

    print("Stage D: raw-complex adapter ingress, axis, precision, and sealed access", flush=True)
    raw = np.empty((16, 16, 1, 600), dtype=np.complex64)
    for tx in range(16):
        for rx in range(16):
            for frequency in range(600):
                raw[tx, rx, 0, frequency] = complex(
                    1000.0 * tx + 10.0 * rx + frequency / 1000.0,
                    -100.0 * tx - rx - frequency / 2000.0,
                )
    averaged = spinr_module._validate_and_average_b787_raw_response(raw)
    gate.check(
        averaged.dtype == np.complex64
        and averaged.shape == (16, 16, 600)
        and averaged[5, 7, 11] == raw[5, 7, 0, 11],
        "raw complex64 ingress preserves real and imaginary channel/frequency values while averaging only chirps",
    )
    try:
        spinr_module._validate_and_average_b787_raw_response(raw.real.astype(np.float32))
    except ValueError:
        wrong_raw_dtype_rejected = True
    else:
        wrong_raw_dtype_rejected = False
    gate.check(wrong_raw_dtype_rejected, "raw adapter ingress refuses non-complex64 source data")

    rendered = npz_tx_rx_frequency_to_renderer(
        torch.as_tensor(averaged, dtype=torch.complex128))
    gate.check(
        rendered.dtype == torch.complex128
        and rendered.shape == (600, 16, 16)
        and rendered[11, 7, 5] == torch.as_tensor(raw[5, 7, 0, 11], dtype=torch.complex128),
        "raw renderer-axis conversion preserves [Tx,Rx,F]→[F,Rx,Tx] without averaging antennas or frequencies",
    )

    # Exercise the unchanged constructor at the actual 3,200/1,000/1,000/4,800
    # role counts without allocating its multi-GiB physical cache.  The raw
    # ingress helper above is independently tested with a real B787-shaped
    # array.  Here the fixture injects two shared, tiny chirp-averaged vectors
    # at that helper and instruments the restricted reader boundary; this lets
    # the constructor itself compute (rather than receive) its train-only
    # normalization statistic.
    train_ids = tuple(range(3200))
    validation_ids = tuple(range(3200, 4200))
    reserved_test_ids = tuple(range(4200, 5200))
    unused_ids = tuple(range(5200, 10000))
    authorized_ids = train_ids + validation_ids
    sealed_contract: Mapping[str, Any] = {
        "role_ids": {
            "train": list(train_ids),
            "validation": list(validation_ids),
            "reserved_test": list(reserved_test_ids),
            "unused": list(unused_ids),
        },
        "response_access": {
            "reserved_test_materialized": False,
            "unused_materialized": False,
        },
        "response_shape": [10000, 16, 16, 1, 600],
    }
    arrays: Mapping[str, object] = {
        "response": None,
        "_lazy_response_reader": object(),
        "meta": {
            "target_type": "b787",
            "experiment": "sphere10k",
            "radar_fc_hz": 10.0e9,
            "radar_bandwidth_hz": 3.0e9,
            "num_adc_samples": 600,
        },
        "rx_pos": np.zeros((10000, 16, 3), dtype=np.float64),
        "tx_pos": np.zeros((10000, 16, 3), dtype=np.float64),
    }
    train_marker = object()
    validation_marker = object()
    train_signal = np.asarray([1.0 + 2.0j, 3.0 + 4.0j], dtype=np.complex64)
    validation_signal = np.asarray([1000.0 + 0.0j, 2000.0 + 0.0j], dtype=np.complex64)
    restricted_requests: list[tuple[int, ...]] = []
    iterator_requests: list[tuple[int, ...]] = []
    yielded_ids: list[int] = []
    original_restrict = spinr_module.restrict_npz_response_views
    original_iter = spinr_module.iter_npz_response_views
    original_ingress = spinr_module._validate_and_average_b787_raw_response

    def fake_restrict(source_arrays: dict[str, object], requested_ids: Sequence[int]) -> dict[str, object]:
        restricted_requests.append(tuple(int(item) for item in requested_ids))
        return {"fixture": source_arrays}

    def fake_iter(_restricted: dict[str, object], requested_ids: Sequence[int]) -> Any:
        requested = tuple(int(item) for item in requested_ids)
        iterator_requests.append(requested)
        for source_id in requested:
            yielded_ids.append(source_id)
            yield source_id, train_marker if source_id in train_ids else validation_marker

    def fake_ingress(marker: object) -> np.ndarray:
        if marker is train_marker:
            return train_signal
        if marker is validation_marker:
            return validation_signal
        raise AssertionError("constructor fixture received an unrecognized raw source view")

    spinr_module.restrict_npz_response_views = fake_restrict
    spinr_module.iter_npz_response_views = fake_iter
    spinr_module._validate_and_average_b787_raw_response = fake_ingress
    try:
        views = SealedRawComplexViews(arrays, sealed_contract)
    finally:
        spinr_module.restrict_npz_response_views = original_restrict
        spinr_module.iter_npz_response_views = original_iter
        spinr_module._validate_and_average_b787_raw_response = original_ingress

    expected_train_power = 15.0
    validation_power = float(np.vdot(validation_signal, validation_signal).real / validation_signal.size)
    gate.check(
        restricted_requests == [authorized_ids]
        and iterator_requests == [authorized_ids]
        and yielded_ids == list(authorized_ids)
        and set(views._views) == set(authorized_ids)
        and views.role_ids("train") == train_ids
        and views.role_ids("validation") == validation_ids
        and views.view(train_ids[0]).rx_pos_m.dtype == np.float64
        and views.view(train_ids[0]).tx_pos_m.dtype == np.float64,
        "actual constructor requests, yields, and caches exactly canonical authorized IDs with FP64 poses",
    )
    denied = []
    for sealed_id in (reserved_test_ids[0], unused_ids[0]):
        for method in (lambda source_id=sealed_id: views.view(source_id),
                       lambda source_id=sealed_id: views.tensor_view(source_id, device="cpu")):
            try:
                method()
            except PermissionError:
                denied.append(sealed_id)
    gate.check(
        denied == [reserved_test_ids[0], reserved_test_ids[0], unused_ids[0], unused_ids[0]],
        "both reserved-test and unused IDs are denied by view and tensor_view",
    )
    gate.check(
        math.isclose(views.raw_training_mean_power(), expected_train_power, rel_tol=0.0, abs_tol=0.0)
        and not math.isclose(views.raw_training_mean_power(), validation_power, rel_tol=1e-6),
        "actual constructor computes train-only mean power 15 despite much larger validation energy",
    )


def stage_product_renderer_and_real_vjp(gate: Gate) -> None:
    print("Stage D: independent product renderer, adjoint, and real-field VJP", flush=True)
    device = torch.device("cpu")
    frequencies, kvector, rx_pos, tx_pos, points, weights = _tiny_physics(device)
    reference = _direct_product_forward(
        frequencies, kvector, rx_pos, tx_pos, points, weights, phase_sign=SPINR_STYLE_PHASE_SIGN)
    accelerated = range_forward_operator(
        frequencies, kvector, rx_pos, tx_pos, points, weights,
        phase_sign=SPINR_STYLE_PHASE_SIGN, range_model=SPINR_STYLE_RANGE_MODEL,
        pair_chunk=2, point_chunk=2, compute_dtype=torch.float64,
    )
    gate.near(accelerated, reference, 1e-8, "accelerated product/-1 renderer matches independent complex128 reference")
    monostatic_position = torch.tensor([[1.85, 0.0, 0.0]], dtype=torch.float64)
    sum2 = range_forward_operator(
        frequencies, kvector, monostatic_position, monostatic_position, points, weights,
        phase_sign=SPINR_STYLE_PHASE_SIGN, range_model="sum2",
        pair_chunk=1, point_chunk=2, compute_dtype=torch.float64,
    )
    product = range_forward_operator(
        frequencies, kvector, monostatic_position, monostatic_position, points, weights,
        phase_sign=SPINR_STYLE_PHASE_SIGN, range_model="product",
        pair_chunk=1, point_chunk=2, compute_dtype=torch.float64,
    )
    ratio = float((product / sum2).abs().mean())
    gate.check(abs(ratio - 4.0) < 1e-7, "product spreading is four times sum2 in a monostatic geometry")
    torch.manual_seed(19)
    residual = torch.randn_like(accelerated) + 1j * torch.randn_like(accelerated)
    adjoint = range_adjoint_operator(
        frequencies, kvector, rx_pos, tx_pos, points, residual,
        phase_sign=SPINR_STYLE_PHASE_SIGN, range_model=SPINR_STYLE_RANGE_MODEL,
        pair_chunk=2, point_chunk=2, compute_dtype=torch.float64,
    )
    direct_adjoint = _direct_product_adjoint(
        frequencies, kvector, rx_pos, tx_pos, points, residual, phase_sign=SPINR_STYLE_PHASE_SIGN)
    gate.near(adjoint, direct_adjoint, 1e-8, "accelerated product adjoint matches independent reference")
    gate.near(torch.vdot(accelerated.reshape(-1), residual.reshape(-1)), torch.vdot(weights, adjoint), 1e-8,
              "product forward/adjoint obey the complex dot identity")

    q = torch.randn(points.shape[0], dtype=torch.float64, requires_grad=True)
    volume = 0.003
    a_init = 0.7
    scaled_weights = scale_field_to_renderer_weights(q, cell_volume_m3=volume, initial_output_scale=a_init)
    predicted = range_forward_operator(
        frequencies, kvector, rx_pos, tx_pos, points, scaled_weights,
        phase_sign=SPINR_STYLE_PHASE_SIGN, range_model=SPINR_STYLE_RANGE_MODEL,
        pair_chunk=2, point_chunk=2, compute_dtype=torch.float64,
    )
    observed = 0.2 * reference
    power = float(observed.abs().square().mean())
    autograd_loss = spinr_style_objective(predicted, observed, training_mean_raw_power=power)
    autograd_loss.backward()
    response_loss, response_grad = spinr_train.response_cotangent(
        predicted.detach(), observed, training_mean_raw_power=power)
    manual = spinr_train.real_field_cotangent_from_response(
        response_cotangent_frequency=response_grad,
        frequencies_hz=frequencies,
        kvector=kvector,
        rx_pos_m=rx_pos,
        tx_pos_m=tx_pos,
        points_m=points,
        cell_volume_m3=volume,
        initial_output_scale=a_init,
        renderer_point_tile=2,
        pair_tile=2,
    )
    gate.check(abs(float(response_loss - autograd_loss.detach())) < 1e-12, "response cotangent uses the native all-bin loss")
    gate.near(manual, q.grad, 2e-7, "manual scale*Re(A^H g) equals real autograd field gradient without extra factor")


class _TinyViews:
    def __init__(self, entries: Mapping[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> None:
        self.entries = dict(entries)

    def tensor_view(self, source_id: int, *, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        observed, rx, tx = self.entries[int(source_id)]
        return observed.to(device=device), rx.to(device=device), tx.to(device=device)


def _canonical_contract_and_header_arrays() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build metadata-only canonical B787 fixtures; no response array exists."""

    roles = {
        "train": list(range(3200)),
        "validation": list(range(3200, 4200)),
        "reserved_test": list(range(4200, 5200)),
        "unused": list(range(5200, 10000)),
    }
    contract: dict[str, Any] = {
        "schema": "rift_npz_sealed_protocol_v1",
        "version": 1,
        "data_format": "npz",
        "source_path": "/temporary/archive-a/b787.npz",
        "response_shape": [10000, 16, 16, 1, 600],
        "response_dtype": "complex64",
        "role_manifest_path": "/temporary/archive-a/split.json",
        "role_manifest_name": "canonical",
        "split_strategy": "synthetic-canonical",
        "role_ids": roles,
        "response_access": {
            "train_materialized": True,
            "validation_materialized": True,
            "reserved_test_materialized": False,
            "unused_materialized": False,
        },
    }
    # About 7.7 MiB total, deliberately far smaller than a signal payload and
    # sufficient to make a one-bit pose mismatch observable in the identity.
    tx_pos = np.zeros((10000, 16, 3), dtype=np.float64)
    rx_pos = np.zeros((10000, 16, 3), dtype=np.float64)
    tx_pos[:, :, 0] = 1.8
    rx_pos[:, :, 0] = 1.9
    arrays: dict[str, Any] = {
        "response": None,
        "meta": {
            "target_type": "b787",
            "experiment": "sphere10k",
            "radar_fc_hz": 10.0e9,
            "radar_bandwidth_hz": 3.0e9,
            "num_adc_samples": 600,
        },
        "tx_pos": tx_pos,
        "rx_pos": rx_pos,
    }
    acquisition = build_spinr_style_acquisition_identity(arrays, contract)
    return contract, arrays, acquisition


def _cpu_run_args(
    *,
    checkpoint_root: Path,
    checkpoint_name: str,
    resume: str | None,
    epochs: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        npz_path="/must-not-be-opened/b787.npz",
        npz_role_manifest="/must-not-be-opened/split.json",
        checkpoint_root=str(checkpoint_root),
        checkpoint_name=checkpoint_name,
        epochs=int(epochs),
        grid_size=48,
        device="cpu",
        allow_cpu_validation=True,
        neural_point_tile=2,
        renderer_point_tile=2,
        pair_tile=1,
        host_rss_limit_gib=64.0,
        resume=resume,
    )


def stage_preflight_namespace_and_acquisition(gate: Gate) -> None:
    """Exercise the actual entrypoint's payload-free rejection ordering."""

    print("Stage E: pre-payload recipe, namespace, and acquisition identity gates", flush=True)
    contract, arrays, acquisition_identity = _canonical_contract_and_header_arrays()
    roles = contract["role_ids"]
    model = SpinrStyleINR()
    optimizer = Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
    scheduler = CosineAnnealingLR(optimizer, T_max=300, eta_min=1e-5)
    recipe = spinr_train._recipe_identity()
    baseline_state = spinr_train._checkpoint_state(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        sealed_contract=contract,
        acquisition_identity=acquisition_identity,
        recipe_identity=recipe,
        operational_settings={"device": "cpu", "neural_point_tile": 2, "renderer_point_tile": 2, "pair_tile": 1},
        training_mean_raw_power=1.0,
        initial_output_scale=0.1,
        initial_scale_ids=roles["train"][:32],
        initial_scale_observed_energy=1.0,
        initial_scale_predicted_energy=1.0,
        epoch_index=0,
        execution=spinr_train._execution_state(
            phase="updates", completed_updates=0, loss_sum=0.0, gradient_norm_sum=0.0, elapsed_seconds=0.0),
        best_validation_rel_mse=math.inf,
        best_epoch=None,
        history=[],
    )
    relocated_contract = dict(contract)
    relocated_contract["source_path"] = "/different/mount/b787.npz"
    relocated_contract["role_manifest_path"] = "/different/mount/split.json"
    relocated_identity = build_spinr_style_acquisition_identity(arrays, relocated_contract)
    validate_spinr_style_acquisition_identity(acquisition_identity, relocated_identity)
    gate.check(True, "path relocation preserves the path-free acquisition identity")

    changed_arrays = dict(arrays)
    changed_tx = np.array(arrays["tx_pos"], copy=True)
    changed_tx[0, 0, 0] += 1.0e-12
    changed_arrays["tx_pos"] = changed_tx
    try:
        validate_spinr_style_acquisition_identity(
            acquisition_identity,
            build_spinr_style_acquisition_identity(changed_arrays, contract),
        )
    except ValueError:
        changed_pose_rejected = True
    else:
        changed_pose_rejected = False
    gate.check(changed_pose_rejected, "one-bit authorized pose change rejects before any raw response reader")

    changed_meta_arrays = dict(arrays)
    changed_meta_arrays["meta"] = {**arrays["meta"], "radar_fc_hz": 10.0e9 + 0.5}
    try:
        validate_spinr_style_acquisition_identity(
            acquisition_identity,
            build_spinr_style_acquisition_identity(changed_meta_arrays, contract),
        )
    except ValueError:
        changed_metadata_rejected = True
    else:
        changed_metadata_rejected = False
    changed_frequency = dict(acquisition_identity)
    changed_frequency["frequency_hz"] = acquisition_identity["frequency_hz"].clone()
    changed_frequency["frequency_hz"][0] += 0.5
    try:
        validate_spinr_style_acquisition_identity(changed_frequency, acquisition_identity)
    except ValueError:
        changed_frequency_rejected = True
    else:
        changed_frequency_rejected = False
    gate.check(
        changed_metadata_rejected and changed_frequency_rejected,
        "acquisition identity rejects changed B787 metadata and the derived FP64 frequency grid",
    )

    wrong_precision = dict(arrays)
    wrong_precision["rx_pos"] = np.asarray(arrays["rx_pos"], dtype=np.float32)
    try:
        build_spinr_style_acquisition_identity(wrong_precision, contract)
    except ValueError:
        wrong_precision_rejected = True
    else:
        wrong_precision_rejected = False
    gate.check(wrong_precision_rejected, "source pose precision must already be float64; it is never silently promoted")

    rejected_grids: list[int] = []
    for grid_size in (96, 192):
        grid_args = _cpu_run_args(checkpoint_root=Path("/temporary"), checkpoint_name="grid-check", resume=None)
        grid_args.grid_size = grid_size
        try:
            spinr_train._validate_cli_recipe(grid_args)
        except ValueError:
            rejected_grids.append(grid_size)
    gate.check(rejected_grids == [96, 192], "G=96 and G=192 are rejected by the train entrypoint")

    with tempfile.TemporaryDirectory(prefix="rift_spinr_preflight_") as temporary:
        root = Path(temporary)
        resume_dir = root / "resume"
        resume_dir.mkdir()
        latest = resume_dir / "checkpoint_latest.pth.tar"
        bad_recipe_state = dict(baseline_state)
        bad_recipe_state["spinr_style_recipe"] = {
            **recipe,
            "operator": {**recipe["operator"], "grid_size": 96},
        }
        spinr_train._atomic_torch_save(bad_recipe_state, latest)
        spinr_train._atomic_json_save({"history": []}, resume_dir / "metrics_history.json")
        original_preflight = spinr_train.preflight_b787_development_inputs
        preflight_calls: list[bool] = []

        def payload_sentinel(**_kwargs: Any) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
            preflight_calls.append(True)
            raise AssertionError("entrypoint reached archive preflight after a semantic rejection")

        spinr_train.preflight_b787_development_inputs = payload_sentinel
        try:
            try:
                spinr_train.run(_cpu_run_args(
                    checkpoint_root=root, checkpoint_name="resume", resume=str(latest)))
            except ValueError:
                bad_recipe_rejected = True
            else:
                bad_recipe_rejected = False
        finally:
            spinr_train.preflight_b787_development_inputs = original_preflight
        gate.check(
            bad_recipe_rejected and not preflight_calls,
            "actual run rejects a changed saved recipe before metadata or response access",
        )

        corrupt_dir = root / "corrupt"
        corrupt_dir.mkdir()
        corrupt_state = dict(baseline_state)
        corrupt_state["normalization"] = {
            **baseline_state["normalization"],
            "training_mean_raw_power": 0.0,
        }
        corrupt_latest = corrupt_dir / "checkpoint_latest.pth.tar"
        spinr_train._atomic_torch_save(corrupt_state, corrupt_latest)
        spinr_train._atomic_json_save({"history": []}, corrupt_dir / "metrics_history.json")
        original_preflight = spinr_train.preflight_b787_development_inputs
        corrupt_calls: list[bool] = []

        def corrupt_sentinel(**_kwargs: Any) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
            corrupt_calls.append(True)
            raise AssertionError("entrypoint reached archive preflight after corrupt checkpoint rejection")

        spinr_train.preflight_b787_development_inputs = corrupt_sentinel
        try:
            try:
                spinr_train.run(_cpu_run_args(
                    checkpoint_root=root, checkpoint_name="corrupt", resume=str(corrupt_latest)))
            except ValueError:
                corrupt_rejected = True
            else:
                corrupt_rejected = False
        finally:
            spinr_train.preflight_b787_development_inputs = original_preflight
        gate.check(
            corrupt_rejected and not corrupt_calls,
            "actual run rejects malformed normalization/cursor state before metadata or response access",
        )

        def rejects_before_preflight(name: str, state: Mapping[str, Any]) -> bool:
            directory = root / name
            directory.mkdir()
            checkpoint = directory / "checkpoint_latest.pth.tar"
            spinr_train._atomic_torch_save(state, checkpoint)
            spinr_train._atomic_json_save({"history": state["history"]}, directory / "metrics_history.json")
            original = spinr_train.preflight_b787_development_inputs
            calls: list[bool] = []

            def sentinel(**_kwargs: Any) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
                calls.append(True)
                raise AssertionError("entrypoint reached archive preflight after a checkpoint rejection")

            spinr_train.preflight_b787_development_inputs = sentinel
            try:
                try:
                    spinr_train.run(_cpu_run_args(
                        checkpoint_root=root, checkpoint_name=name, resume=str(checkpoint)))
                except ValueError:
                    return not calls
                return False
            finally:
                spinr_train.preflight_b787_development_inputs = original

        bad_cursor_state = dict(baseline_state)
        bad_cursor_state["execution"] = spinr_train._execution_state(
            phase="updates", completed_updates=1, loss_sum=0.0, gradient_norm_sum=0.0,
            elapsed_seconds=0.0)
        # Keep the saved cursor at zero so the execution object and top-level
        # resume cursor disagree.
        bad_cursor_rejected = rejects_before_preflight("bad-cursor", bad_cursor_state)

        bad_model_state = dict(baseline_state)
        malformed_model = dict(baseline_state["model_state_dict"])
        malformed_model["head.bias"] = torch.zeros((2,), dtype=torch.float32)
        bad_model_state["model_state_dict"] = malformed_model
        bad_model_rejected = rejects_before_preflight("bad-model", bad_model_state)

        bad_model_dtype_state = dict(baseline_state)
        wrong_dtype_model = dict(baseline_state["model_state_dict"])
        wrong_dtype_model["head.bias"] = wrong_dtype_model["head.bias"].to(dtype=torch.float64)
        bad_model_dtype_state["model_state_dict"] = wrong_dtype_model
        bad_model_dtype_rejected = rejects_before_preflight("bad-model-dtype", bad_model_dtype_state)

        completed_state = dict(baseline_state)
        completed_state["execution"] = spinr_train._execution_state(
            phase="completed", completed_updates=0, loss_sum=0.0, gradient_norm_sum=0.0,
            elapsed_seconds=0.0, stop_reason="development_epoch_budget_reached")
        completed_preflight_rejected = rejects_before_preflight("completed-latest", completed_state)
        gate.check(
            bad_cursor_rejected and bad_model_rejected and bad_model_dtype_rejected
            and completed_preflight_rejected,
            "cursor/model-layout/dtype/terminal resumes are rejected before archive metadata or response access",
        )

        identity_dir = root / "identity"
        identity_dir.mkdir()
        identity_latest = identity_dir / "checkpoint_latest.pth.tar"
        spinr_train._atomic_torch_save(baseline_state, identity_latest)
        spinr_train._atomic_json_save({"history": []}, identity_dir / "metrics_history.json")
        changed_identity_arrays = dict(arrays)
        changed_identity_tx = np.array(arrays["tx_pos"], copy=True)
        changed_identity_tx[1, 2, 0] += 1.0e-12
        changed_identity_arrays["tx_pos"] = changed_identity_tx
        changed_identity = build_spinr_style_acquisition_identity(changed_identity_arrays, contract)
        original_preflight = spinr_train.preflight_b787_development_inputs
        original_materialize = spinr_train.materialize_b787_development_views
        raw_materialize_calls: list[bool] = []

        def changed_identity_preflight(**_kwargs: Any) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
            return {}, contract, changed_identity

        def raw_materialize_sentinel(*_args: Any, **_kwargs: Any) -> Any:
            raw_materialize_calls.append(True)
            raise AssertionError("entrypoint reached raw materialization after changed acquisition rejection")

        spinr_train.preflight_b787_development_inputs = changed_identity_preflight
        spinr_train.materialize_b787_development_views = raw_materialize_sentinel
        try:
            try:
                spinr_train.run(_cpu_run_args(
                    checkpoint_root=root, checkpoint_name="identity", resume=str(identity_latest)))
            except ValueError:
                changed_identity_rejected = True
            else:
                changed_identity_rejected = False
        finally:
            spinr_train.preflight_b787_development_inputs = original_preflight
            spinr_train.materialize_b787_development_views = original_materialize
        gate.check(
            changed_identity_rejected and not raw_materialize_calls,
            "actual run rejects changed authorized acquisition poses before raw response materialization",
        )

        original_preflight = spinr_train.preflight_b787_development_inputs
        relocation_calls: list[bool] = []

        def relocation_sentinel(**_kwargs: Any) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
            relocation_calls.append(True)
            raise AssertionError("entrypoint reached archive preflight after output relocation rejection")

        spinr_train.preflight_b787_development_inputs = relocation_sentinel
        try:
            try:
                spinr_train.run(_cpu_run_args(
                    checkpoint_root=root, checkpoint_name="relocated-output", resume=str(identity_latest)))
            except ValueError:
                relocation_rejected = True
            else:
                relocation_rejected = False
        finally:
            spinr_train.preflight_b787_development_inputs = original_preflight
        gate.check(
            relocation_rejected and not relocation_calls,
            "actual run rejects continuation output relocation before archive access",
        )

        repair_dir = root / "repair"
        repair_dir.mkdir()
        repair_latest = repair_dir / "checkpoint_latest.pth.tar"
        spinr_train._atomic_torch_save(baseline_state, repair_latest)
        (repair_dir / "metrics_history.json").write_text("{not-json", encoding="utf-8")
        (repair_dir / ".metrics_history.json.tmp.123").write_text("stale", encoding="utf-8")
        spinr_train._validate_checkpoint_namespace_artifacts(
            checkpoint_dir=repair_dir,
            resume_checkpoint=baseline_state,
            sealed_contract=contract,
            acquisition_identity=acquisition_identity,
            recipe_identity=recipe,
        )
        repaired_metrics = json.loads((repair_dir / "metrics_history.json").read_text(encoding="utf-8"))
        gate.check(
            repaired_metrics == {"history": []}
            and not (repair_dir / ".metrics_history.json.tmp.123").exists(),
            "validated latest checkpoint regenerates an interrupted metrics projection and clears only known private temporaries",
        )

        collision_dir = root / "collision"
        collision_dir.mkdir()
        (collision_dir / "legacy-result.txt").write_text("do not overwrite\n", encoding="utf-8")
        original_preflight = spinr_train.preflight_b787_development_inputs
        collision_calls: list[bool] = []

        def collision_sentinel(**_kwargs: Any) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
            collision_calls.append(True)
            raise AssertionError("entrypoint reached archive preflight after output collision")

        spinr_train.preflight_b787_development_inputs = collision_sentinel
        try:
            try:
                spinr_train.run(_cpu_run_args(
                    checkpoint_root=root, checkpoint_name="collision", resume=None))
            except FileExistsError:
                collision_rejected = True
            else:
                collision_rejected = False
        finally:
            spinr_train.preflight_b787_development_inputs = original_preflight
        gate.check(
            collision_rejected and not collision_calls,
            "actual fresh run rejects a nonempty artifact namespace before archive access or writes",
        )

        original_loader = spinr_train._load_sealed_npz_protocol_contract
        original_builder = spinr_train.build_sealed_raw_complex_views
        raw_builder_calls: list[bool] = []

        def header_only_loader(*_args: Any, **_kwargs: Any) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
            return arrays, contract

        def raw_builder_sentinel(*_args: Any, **_kwargs: Any) -> Any:
            raw_builder_calls.append(True)
            raise AssertionError("raw builder must not be called by metadata-only preflight")

        spinr_train._load_sealed_npz_protocol_contract = header_only_loader
        spinr_train.build_sealed_raw_complex_views = raw_builder_sentinel
        try:
            preflight_arrays, preflight_contract, preflight_identity = spinr_train.preflight_b787_development_inputs(
                npz_path="/relocated/b787.npz",
                manifest_path="/relocated/split.json",
                resume_checkpoint=baseline_state,
            )
            gate.check(
                preflight_arrays is arrays and preflight_contract is contract and not raw_builder_calls,
                "metadata/role/acquisition continuation preflight remains response-payload-free",
            )
            validate_spinr_style_acquisition_identity(acquisition_identity, preflight_identity)
        finally:
            spinr_train._load_sealed_npz_protocol_contract = original_loader
            spinr_train.build_sealed_raw_complex_views = original_builder


def stage_tiled_b4_gradient_and_resume(gate: Gate) -> None:
    print("Stage F: B=4 manual MLP VJP and checkpoint recovery semantics", flush=True)
    device = torch.device("cpu")
    frequencies, kvector, rx_pos, tx_pos, points, _weights = _tiny_physics(device)
    torch.manual_seed(31)
    entries = {}
    for source_id in range(4):
        observed = torch.randn(16, 3, 2, dtype=torch.float64) + 1j * torch.randn(16, 3, 2, dtype=torch.float64)
        entries[source_id] = (observed, rx_pos, tx_pos)
    views = _TinyViews(entries)
    power = float(torch.cat([entry[0].reshape(-1) for entry in entries.values()]).abs().square().mean())
    model_auto = SpinrStyleINR().to(device)
    model_manual = SpinrStyleINR().to(device)
    model_manual.load_state_dict(model_auto.state_dict())
    volume = 0.003
    a_init = 0.7

    q_auto = model_auto(points).to(dtype=torch.float64)
    w_auto = scale_field_to_renderer_weights(q_auto, cell_volume_m3=volume, initial_output_scale=a_init)
    predictions = []
    targets = []
    for source_id in range(4):
        observed, rx, tx = views.tensor_view(source_id, device=device)
        predictions.append(range_forward_operator(
            frequencies, kvector, rx, tx, points, w_auto,
            phase_sign=SPINR_STYLE_PHASE_SIGN, range_model=SPINR_STYLE_RANGE_MODEL,
            pair_chunk=2, point_chunk=2, compute_dtype=torch.float64,
        ))
        targets.append(observed)
    loss_auto = spinr_style_objective(torch.stack(predictions), torch.stack(targets), training_mean_raw_power=power)
    loss_auto.backward()
    auto_grads = {name: parameter.grad.detach().clone() for name, parameter in model_auto.named_parameters()}

    optimizer = Adam(model_manual.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
    loss_manual, _grad_norm = spinr_train.logical_batch_update(
        model=model_manual,
        optimizer=optimizer,
        source_ids=(0, 1, 2, 3),
        views=views,  # type: ignore[arg-type]
        points_m=points,
        cell_volume_m3=volume,
        initial_output_scale=a_init,
        frequencies_hz=frequencies,
        kvector=kvector,
        training_mean_raw_power=power,
        neural_point_tile=2,
        renderer_point_tile=2,
        pair_tile=2,
        device=device,
    )
    gate.check(abs(loss_manual - float(loss_auto.detach())) < 2e-10, "B=4 manual path averages the same native loss")
    for name, parameter in model_manual.named_parameters():
        gate.near(parameter.grad, auto_grads[name], 3e-5, f"B=4 tiled manual VJP matches monolithic gradient for {name}")

    contract, _header_arrays, acquisition_identity = _canonical_contract_and_header_arrays()
    roles = contract["role_ids"]
    scheduler = CosineAnnealingLR(optimizer, T_max=300, eta_min=1e-5)
    recipe = spinr_train._recipe_identity()
    execution = spinr_train._execution_state(
        phase="updates",
        completed_updates=1,
        loss_sum=loss_manual,
        gradient_norm_sum=1.0,
        elapsed_seconds=0.1,
    )
    state = spinr_train._checkpoint_state(
        model=model_manual,
        optimizer=optimizer,
        scheduler=scheduler,
        sealed_contract=contract,
        acquisition_identity=acquisition_identity,
        recipe_identity=recipe,
        operational_settings={"device": "cpu", "neural_point_tile": 2, "renderer_point_tile": 2, "pair_tile": 2},
        training_mean_raw_power=power,
        initial_output_scale=a_init,
        initial_scale_ids=roles["train"][:32],
        initial_scale_observed_energy=2.0,
        initial_scale_predicted_energy=1.0,
        epoch_index=0,
        execution=execution,
        best_validation_rel_mse=math.inf,
        best_epoch=None,
        history=[],
    )
    with tempfile.TemporaryDirectory(prefix="rift_spinr_resume_") as temporary:
        destination = Path(temporary) / "checkpoint_latest.pth.tar"
        spinr_train._atomic_torch_save(state, destination)
        loaded = legacy_train.load_tensor_checkpoint(destination, map_location="cpu")
        restored_model = SpinrStyleINR()
        restored_optimizer = Adam(restored_model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
        restored_scheduler = CosineAnnealingLR(restored_optimizer, T_max=300, eta_min=1e-5)
        restored = spinr_train._restore_checkpoint(
            checkpoint=loaded,
            model=restored_model,
            optimizer=restored_optimizer,
            scheduler=restored_scheduler,
            sealed_contract=contract,
            acquisition_identity=acquisition_identity,
            recipe_identity=recipe,
        )
        gate.check(
            restored["epoch_index"] == 0
            and restored["execution"]["partial_epoch"]["completed_updates"] == 1
            and restored["initial_scale_ids"] == tuple(roles["train"][:32]),
                   "checkpoint restores complete-update cursor and fixed scale IDs")
        gate.check(
            all(torch.equal(parameter, restored_model.state_dict()[name])
                for name, parameter in model_manual.state_dict().items()),
            "checkpoint restores every MLP tensor exactly",
        )
        tampered_optimizer = dict(loaded)
        tampered_optimizer_state = dict(loaded["optimizer_state_dict"])
        tampered_group = dict(tampered_optimizer_state["param_groups"][0])
        tampered_group["lr"] = 0.0
        tampered_group["maximize"] = True
        tampered_optimizer_state["param_groups"] = [tampered_group]
        tampered_optimizer_state["state"] = {}
        tampered_optimizer["optimizer_state_dict"] = tampered_optimizer_state
        try:
            spinr_train._validate_resume_checkpoint_structure(
                tampered_optimizer, recipe_identity=recipe)
        except ValueError:
            optimizer_tamper_rejected = True
        else:
            optimizer_tamper_rejected = False
        gate.check(
            optimizer_tamper_rejected,
            "resume rejects zero-LR/maximize/empty-Adam-state payloads rather than silently changing the recipe",
        )
        backend_tamper = dict(loaded)
        backend_optimizer = dict(loaded["optimizer_state_dict"])
        backend_group = dict(backend_optimizer["param_groups"][0])
        backend_group["foreach"] = True
        backend_optimizer["param_groups"] = [backend_group]
        backend_tamper["optimizer_state_dict"] = backend_optimizer
        try:
            spinr_train._validate_resume_checkpoint_structure(
                backend_tamper, recipe_identity=recipe)
        except ValueError:
            backend_tamper_rejected = True
        else:
            backend_tamper_rejected = False

        moment_tamper = dict(loaded)
        moment_optimizer = dict(loaded["optimizer_state_dict"])
        moment_states = dict(moment_optimizer["state"])
        first_parameter_id = next(iter(moment_states))
        first_moment = dict(moment_states[first_parameter_id])
        first_moment["exp_avg"] = first_moment["exp_avg"].to(dtype=torch.float64)
        moment_states[first_parameter_id] = first_moment
        moment_optimizer["state"] = moment_states
        moment_tamper["optimizer_state_dict"] = moment_optimizer
        try:
            spinr_train._validate_resume_checkpoint_structure(
                moment_tamper, recipe_identity=recipe)
        except ValueError:
            moment_tamper_rejected = True
        else:
            moment_tamper_rejected = False
        gate.check(
            backend_tamper_rejected and moment_tamper_rejected,
            "resume rejects an Adam backend override or FP64 moment that would silently change the FP32 trajectory",
        )
        try:
            spinr_train._restore_checkpoint(
                checkpoint=loaded,
                model=SpinrStyleINR(),
                optimizer=Adam(SpinrStyleINR().parameters(), lr=1e-4),
                scheduler=CosineAnnealingLR(Adam(SpinrStyleINR().parameters(), lr=1e-4), T_max=300),
                sealed_contract=contract,
                acquisition_identity=acquisition_identity,
                recipe_identity={
                    **recipe,
                    "operator": {**recipe["operator"], "grid_size": 96},
                },
            )
        except ValueError:
            mismatch_rejected = True
        else:
            mismatch_rejected = False
        gate.check(mismatch_rejected, "resume rejects a changed scientific quadrature recipe before payload access")


def stage_actual_run_resume_state_machine(gate: Gate) -> None:
    """Run the real entrypoint state machine with tiny deterministic harness hooks.

    The harness preserves the production Adam, scheduler, checkpoint, signal,
    and resume paths while replacing only the expensive renderer/data adapter.
    Its nonzero-LR all-parameter update makes interrupted-versus-uninterrupted equality
    meaningful without pretending that a two-update fixture is a B787 result.
    """

    print("Stage G: actual entrypoint interruption/resume state-machine harness", flush=True)
    contract, _arrays, acquisition_identity = _canonical_contract_and_header_arrays()

    class HarnessViews:
        frequencies_hz = np.linspace(8.5e9, 11.495e9, 600, dtype=np.float64)
        materialized_response_bytes = 0

        @staticmethod
        def role_ids(role: str) -> tuple[int, ...]:
            if role == "train":
                return tuple(range(8))
            if role == "validation":
                return (8, 9)
            raise ValueError(role)

        @staticmethod
        def raw_training_mean_power() -> float:
            return 1.0

    class HarnessStopper:
        def __init__(self) -> None:
            self.requested = False
            stopper_box[0] = self

        def __call__(self, _signum: int, _frame: Any) -> None:
            self.requested = True

    original_values = {
        name: getattr(spinr_train, name)
        for name in (
            "CANONICAL_TRAIN_COUNT",
            "CANONICAL_VALIDATION_COUNT",
            "CANONICAL_UPDATES_PER_EPOCH",
            "CANONICAL_VIEW_BATCH",
            "CANONICAL_INIT_SCALE_COUNT",
            "CANONICAL_TRAIN_DIAGNOSTIC_COUNT",
            "CANONICAL_MIN_EPOCHS",
            "CANONICAL_VALIDATION_EVERY",
            "preflight_b787_development_inputs",
            "materialize_b787_development_views",
            "midpoint_grid",
            "estimate_initial_output_scale",
            "logical_batch_update",
            "evaluate_role",
            "_enforce_memory_gates",
            "_StopAfterCurrentUpdate",
            "_save_checkpoint",
            "_save_latest_and_history",
            "plateau_reached",
        )
    }
    stopper_box: list[HarnessStopper | None] = [None]
    mode: dict[str, Any] = {
        "interrupt_after_update": None,
        "signal_during_validation": False,
        "crash_after_pending_epoch": None,
        "crash_after_committed_epoch": None,
        "crash_after_artifact": None,
        "calls_this_run": 0,
        "trace": [],
        "materialize_calls": 0,
    }

    class SimulatedProcessLoss(RuntimeError):
        pass

    def fake_preflight(**_kwargs: Any) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
        return {}, contract, acquisition_identity

    def fake_materialize(*_args: Any, **_kwargs: Any) -> HarnessViews:
        mode["materialize_calls"] += 1
        return HarnessViews()

    def fake_midpoint_grid(*_args: Any, **_kwargs: Any) -> tuple[torch.Tensor, float]:
        return torch.zeros((1, 3), dtype=torch.float64), 1.0

    def fake_initial_scale(**_kwargs: Any) -> tuple[float, tuple[int, ...], float, float]:
        return 0.1, (0, 1), 1.0, 1.0

    def fake_logical_batch_update(*, model: SpinrStyleINR, optimizer: Adam, source_ids: Sequence[int], **_kwargs: Any) -> tuple[float, float]:
        optimizer.zero_grad(set_to_none=True)
        # A real nonzero-LR Adam update on every trainable tensor.  The fixed
        # B787 model remains intact; only renderer work is replaced by the
        # fixture, so the strict optimizer-state recovery contract is tested.
        random_scale = 1.0 + 0.01 * (
            random.random() + float(np.random.random()) + float(torch.rand((), dtype=torch.float32)))
        loss_tensor = random_scale * sum(parameter.square().mean() for parameter in model.parameters())
        loss_tensor.backward()
        # Report an above-threshold *unclipped* norm so the production state
        # machine must preserve the clipping-rate accumulator through a
        # checkpoint.  The real parameter update above remains a nonzero-LR
        # Adam update over every tensor.
        grad_norm = 1.5
        optimizer.step()
        mode["trace"].append(tuple(int(item) for item in source_ids))
        mode["calls_this_run"] += 1
        trigger_after = mode["interrupt_after_update"]
        if trigger_after is not None and mode["calls_this_run"] == int(trigger_after):
            if stopper_box[0] is None:
                raise AssertionError("harness update could not find the active signal stopper")
            stopper_box[0].requested = True
        return float(loss_tensor.detach().item()), grad_norm

    def fake_evaluate_role(*, role: str, source_ids: Sequence[int] | None, **_kwargs: Any) -> dict[str, float | int]:
        if mode["signal_during_validation"] and role == "validation":
            if stopper_box[0] is None:
                raise AssertionError("harness validation could not find the active signal stopper")
            stopper_box[0].requested = True
        relative_mse = 0.20 if role == "train" else 0.10
        return {
            "views": len(source_ids) if source_ids is not None else 2,
            "coherent_relative_mse": relative_mse,
            "coherent_relative_l2": math.sqrt(relative_mse),
            "native_spectral_objective": 0.0,
        }

    plateau_history = [
        {
            "epoch": epoch,
            "running_best_train_diagnostic_rel_mse": value,
            "running_best_validation_rel_mse": value,
        }
        for epoch, value in ((120, 1.0), (130, 0.995), (140, 0.990), (150, 0.985))
    ]
    gate.check(
        spinr_train.plateau_reached(plateau_history),
        "plateau rule accepts exactly three consecutive sub-1% 10-epoch windows at epoch 150",
    )
    boundary_history = [dict(record) for record in plateau_history]
    boundary_history[1]["running_best_train_diagnostic_rel_mse"] = 0.99
    boundary_history[1]["running_best_validation_rel_mse"] = 0.99
    gate.check(
        not spinr_train.plateau_reached(boundary_history),
        "a full 1% improvement at a required adjacent 10-epoch window prevents plateau stopping",
    )

    def fault_inject_save_latest_and_history(**kwargs: Any) -> None:
        original_values["_save_latest_and_history"](**kwargs)
        state = kwargs["state"]
        if (
                mode["crash_after_pending_epoch"] == state["epoch_index"]
                and state["execution"]["phase"] == "pending_epoch_finalization"):
            raise SimulatedProcessLoss("simulated process loss after durable pending checkpoint")
        if (
                mode["crash_after_committed_epoch"] == state["epoch_index"]
                and state["execution"]["phase"] == "updates"
                and len(state["history"]) == state["epoch_index"]):
            raise SimulatedProcessLoss("simulated process loss after committed latest/history")

    def fault_inject_save_checkpoint(**kwargs: Any) -> None:
        original_values["_save_checkpoint"](**kwargs)
        if mode["crash_after_artifact"] == kwargs["checkpoint_name"]:
            raise SimulatedProcessLoss(
                f"simulated process loss after durable {kwargs['checkpoint_name']}")

    try:
        spinr_train.CANONICAL_TRAIN_COUNT = 8
        spinr_train.CANONICAL_VALIDATION_COUNT = 2
        spinr_train.CANONICAL_UPDATES_PER_EPOCH = 2
        spinr_train.CANONICAL_VIEW_BATCH = 4
        spinr_train.CANONICAL_INIT_SCALE_COUNT = 2
        spinr_train.CANONICAL_TRAIN_DIAGNOSTIC_COUNT = 2
        spinr_train.CANONICAL_VALIDATION_EVERY = 1
        # Exercise the epoch-150 artifact branch at fixture epoch one; this is
        # a state-machine gate only, not a claim about a real 150-epoch result.
        spinr_train.CANONICAL_MIN_EPOCHS = 1
        spinr_train.preflight_b787_development_inputs = fake_preflight
        spinr_train.materialize_b787_development_views = fake_materialize
        spinr_train.midpoint_grid = fake_midpoint_grid
        spinr_train.estimate_initial_output_scale = fake_initial_scale
        spinr_train.logical_batch_update = fake_logical_batch_update
        spinr_train.evaluate_role = fake_evaluate_role
        spinr_train._enforce_memory_gates = lambda **_kwargs: {}
        spinr_train._StopAfterCurrentUpdate = HarnessStopper
        spinr_train._save_checkpoint = fault_inject_save_checkpoint
        spinr_train._save_latest_and_history = fault_inject_save_latest_and_history

        with tempfile.TemporaryDirectory(prefix="rift_spinr_state_machine_") as temporary:
            root = Path(temporary)

            def invoke(name: str, *, resume: str | None, interrupt_after_update: int | None,
                       signal_during_validation: bool, crash_after_pending_epoch: int | None = None,
                       crash_after_committed_epoch: int | None = None,
                       crash_after_artifact: str | None = None,
                       epochs: int = 1) -> list[tuple[int, ...]]:
                mode["interrupt_after_update"] = interrupt_after_update
                mode["signal_during_validation"] = signal_during_validation
                mode["crash_after_pending_epoch"] = crash_after_pending_epoch
                mode["crash_after_committed_epoch"] = crash_after_committed_epoch
                mode["crash_after_artifact"] = crash_after_artifact
                mode["calls_this_run"] = 0
                mode["trace"] = []
                mode["materialize_calls"] = 0
                stopper_box[0] = None
                spinr_train.run(_cpu_run_args(
                    checkpoint_root=root, checkpoint_name=name, resume=resume, epochs=epochs))
                return list(mode["trace"])

            baseline_trace = invoke(
                "baseline", resume=None, interrupt_after_update=None, signal_during_validation=False)
            baseline_final = legacy_train.load_tensor_checkpoint(
                root / "baseline" / "checkpoint_final.pth.tar", map_location="cpu")

            mid_trace_first = invoke(
                "mid", resume=None, interrupt_after_update=1, signal_during_validation=False)
            mid_latest = root / "mid" / "checkpoint_latest.pth.tar"
            mid_state = legacy_train.load_tensor_checkpoint(mid_latest, map_location="cpu")
            gate.check(
                mid_state["execution"]["phase"] == "updates"
                and mid_state["execution"]["partial_epoch"]["completed_updates"] == 1,
                "mid-epoch TERM saves a complete nonzero-LR update with partial accumulators",
            )
            gate.check(
                mid_state["execution"]["partial_epoch"]["clipped_update_count"] == 1,
                "mid-epoch recovery retains the unclipped-norm clipping-rate accumulator",
            )
            mid_trace_second = invoke(
                "mid", resume=str(mid_latest), interrupt_after_update=None, signal_during_validation=False)
            mid_final = legacy_train.load_tensor_checkpoint(root / "mid" / "checkpoint_final.pth.tar", map_location="cpu")
            gate.check(
                baseline_trace == mid_trace_first + mid_trace_second,
                "resumed next update uses the same deterministic four-view batch as uninterrupted training",
            )
            gate.check(
                all(torch.equal(value, mid_final["model_state_dict"][name])
                    for name, value in baseline_final["model_state_dict"].items()),
                "nonzero-LR interrupted/resumed model equals uninterrupted model exactly",
            )
            baseline_history = [
                {key: value for key, value in record.items() if key != "epoch_seconds"}
                for record in baseline_final["history"]
            ]
            mid_history = [
                {key: value for key, value in record.items() if key != "epoch_seconds"}
                for record in mid_final["history"]
            ]
            gate.check(
                _tree_equal(baseline_final["optimizer_state_dict"], mid_final["optimizer_state_dict"])
                and _tree_equal(baseline_final["scheduler_state_dict"], mid_final["scheduler_state_dict"])
                and _tree_equal(baseline_final["rng_state"], mid_final["rng_state"])
                and _tree_equal(baseline_final["selection"], mid_final["selection"])
                and _tree_equal(baseline_history, mid_history),
                "interrupted/resumed run preserves Adam, scheduler, RNG, selection, and non-wall-clock history exactly",
            )

            try:
                invoke(
                    "pending", resume=None, interrupt_after_update=None,
                    signal_during_validation=False, crash_after_pending_epoch=0)
            except SimulatedProcessLoss:
                pending_crash_observed = True
            else:
                pending_crash_observed = False
            pending_dir = root / "pending"
            pending_latest = legacy_train.load_tensor_checkpoint(
                pending_dir / "checkpoint_latest.pth.tar", map_location="cpu")
            gate.check(
                pending_crash_observed
                and pending_latest["execution"]["phase"] == "pending_epoch_finalization"
                and pending_latest["execution"]["partial_epoch"]["completed_updates"] == 2
                and pending_latest["execution"]["partial_epoch"]["clipped_update_count"] == 2
                and pending_latest["scheduler_state_dict"]["last_epoch"] == 0,
                "hard loss after update 800 preserves the pre-scheduler pending-finalization checkpoint",
            )
            pending_resume_trace = invoke(
                "pending", resume=str(pending_dir / "checkpoint_latest.pth.tar"),
                interrupt_after_update=None, signal_during_validation=False)
            pending_final = legacy_train.load_tensor_checkpoint(
                pending_dir / "checkpoint_final.pth.tar", map_location="cpu")
            gate.check(
                not pending_resume_trace
                and all(torch.equal(value, pending_final["model_state_dict"][name])
                        for name, value in baseline_final["model_state_dict"].items()),
                "pending-finalization resume replays scheduler/validation once without a duplicate update",
            )
            pending_history = [
                {key: value for key, value in record.items() if key != "epoch_seconds"}
                for record in pending_final["history"]
            ]
            gate.check(
                _tree_equal(pending_final["optimizer_state_dict"], baseline_final["optimizer_state_dict"])
                and _tree_equal(pending_final["scheduler_state_dict"], baseline_final["scheduler_state_dict"])
                and _tree_equal(pending_final["rng_state"], baseline_final["rng_state"])
                and _tree_equal(pending_final["selection"], baseline_final["selection"])
                and _tree_equal(pending_history, baseline_history),
                "pending-finalization recovery takes exactly the missing scheduler step and preserves selection/RNG/history",
            )

            try:
                invoke(
                    "ahead", resume=None, interrupt_after_update=None,
                    signal_during_validation=False,
                    crash_after_artifact="checkpoint_best.pth.tar")
            except SimulatedProcessLoss:
                ahead_crash_observed = True
            else:
                ahead_crash_observed = False
            ahead_dir = root / "ahead"
            ahead_latest = legacy_train.load_tensor_checkpoint(
                ahead_dir / "checkpoint_latest.pth.tar", map_location="cpu")
            ahead_best = legacy_train.load_tensor_checkpoint(
                ahead_dir / "checkpoint_best.pth.tar", map_location="cpu")
            gate.check(
                ahead_crash_observed
                and ahead_latest["execution"]["phase"] == "pending_epoch_finalization"
                and ahead_best["epoch_index"] == ahead_latest["epoch_index"] + 1,
                "crash after selected-artifact write leaves the sole permitted one-epoch-ahead recovery window",
            )
            ahead_resume_trace = invoke(
                "ahead", resume=str(ahead_dir / "checkpoint_latest.pth.tar"),
                interrupt_after_update=None, signal_during_validation=False)
            ahead_final = legacy_train.load_tensor_checkpoint(
                ahead_dir / "checkpoint_final.pth.tar", map_location="cpu")
            gate.check(
                not ahead_resume_trace
                and _tree_equal(ahead_final["scheduler_state_dict"], baseline_final["scheduler_state_dict"])
                and _tree_equal(ahead_final["selection"], baseline_final["selection"]),
                "exact one-epoch-ahead selected artifact is reconciled by replaying finalization once",
            )

            ahead_bad_dir = root / "ahead-bad"
            ahead_bad_dir.mkdir()
            spinr_train._atomic_torch_save(ahead_latest, ahead_bad_dir / "checkpoint_latest.pth.tar")
            spinr_train._atomic_json_save({"history": []}, ahead_bad_dir / "metrics_history.json")
            altered_best = dict(ahead_best)
            altered_model = dict(ahead_best["model_state_dict"])
            altered_head = altered_model["head.bias"].clone()
            altered_head[0] += 1.0
            altered_model["head.bias"] = altered_head
            altered_best["model_state_dict"] = altered_model
            spinr_train._atomic_torch_save(altered_best, ahead_bad_dir / "checkpoint_best.pth.tar")
            try:
                spinr_train._validate_checkpoint_namespace_artifacts(
                    checkpoint_dir=ahead_bad_dir,
                    resume_checkpoint=ahead_latest,
                    sealed_contract=contract,
                    acquisition_identity=acquisition_identity,
                    recipe_identity=spinr_train._recipe_identity(),
                )
            except ValueError:
                ahead_mismatch_rejected = True
            else:
                ahead_mismatch_rejected = False
            gate.check(
                ahead_mismatch_rejected,
                "a structurally valid but divergent future selected artifact is rejected rather than treated as recoverable",
            )

            def copy_ahead_window(name: str, future_best: Mapping[str, Any]) -> Path:
                destination = root / name
                destination.mkdir()
                spinr_train._atomic_torch_save(ahead_latest, destination / "checkpoint_latest.pth.tar")
                spinr_train._atomic_torch_save(future_best, destination / "checkpoint_best.pth.tar")
                spinr_train._atomic_json_save({"history": []}, destination / "metrics_history.json")
                return destination

            aggregate_field_rejections: list[bool] = []
            aggregate_replacements = {
                "streamed_train_native_objective": (
                    float(ahead_best["history"][-1]["streamed_train_native_objective"]) + 1.0),
                "mean_unclipped_gradient_norm": (
                    float(ahead_best["history"][-1]["mean_unclipped_gradient_norm"]) + 1.0),
                "fraction_clipped_updates": 0.5,
                "epoch_seconds": float(ahead_best["history"][-1]["epoch_seconds"]) + 1.0,
            }
            for field, replacement in aggregate_replacements.items():
                altered_aggregate = dict(ahead_best)
                altered_history = list(ahead_best["history"])
                altered_record = dict(altered_history[-1])
                altered_record[field] = replacement
                altered_history[-1] = altered_record
                altered_aggregate["history"] = altered_history
                aggregate_dir = copy_ahead_window(f"ahead-aggregate-{field}", altered_aggregate)
                try:
                    spinr_train._validate_checkpoint_namespace_artifacts(
                        checkpoint_dir=aggregate_dir,
                        resume_checkpoint=ahead_latest,
                        sealed_contract=contract,
                        acquisition_identity=acquisition_identity,
                        recipe_identity=spinr_train._recipe_identity(),
                    )
                except ValueError:
                    aggregate_field_rejections.append(True)
                else:
                    aggregate_field_rejections.append(False)
            gate.check(
                all(aggregate_field_rejections),
                "each ahead-artifact streamed aggregate is bound to the durable pending accumulators before raw materialization",
            )

            altered_evaluation = dict(ahead_best)
            altered_evaluation_history = list(ahead_best["history"])
            altered_evaluation_record = dict(altered_evaluation_history[-1])
            altered_validation = dict(altered_evaluation_record["validation"])
            altered_validation["coherent_relative_mse"] = 0.11
            altered_validation["coherent_relative_l2"] = math.sqrt(0.11)
            altered_evaluation_record["validation"] = altered_validation
            altered_evaluation_record["running_best_validation_rel_mse"] = 0.11
            altered_evaluation_history[-1] = altered_evaluation_record
            altered_evaluation["history"] = altered_evaluation_history
            altered_evaluation["selection"] = {
                **ahead_best["selection"],
                "best_validation_rel_mse": 0.11,
            }
            evaluation_mismatch_dir = copy_ahead_window("ahead-evaluation-mismatch", altered_evaluation)
            try:
                invoke(
                    "ahead-evaluation-mismatch",
                    resume=str(evaluation_mismatch_dir / "checkpoint_latest.pth.tar"),
                    interrupt_after_update=None, signal_during_validation=False)
            except ValueError:
                evaluation_mismatch_rejected = True
            else:
                evaluation_mismatch_rejected = False
            gate.check(
                evaluation_mismatch_rejected and mode["materialize_calls"] == 1,
                "ahead validation evidence must match the replayed authorized finalization before it can survive",
            )

            validation_trace = invoke(
                "validation", resume=None, interrupt_after_update=None,
                signal_during_validation=True, epochs=2)
            validation_dir = root / "validation"
            validation_latest = legacy_train.load_tensor_checkpoint(
                validation_dir / "checkpoint_latest.pth.tar", map_location="cpu")
            gate.check(
                validation_latest["execution"]["phase"] == "updates"
                and validation_latest["epoch_index"] == 1
                and len(validation_latest["history"]) == 1
                and "validation" in validation_latest["history"][0],
                "TERM at update 800/validation commits scheduler, validation, and history before exiting",
            )
            gate.check(
                validation_latest["history"][0]["fraction_clipped_updates"] == 1.0,
                "committed epoch logs the fraction of updates whose unclipped norm exceeded the clip threshold",
            )
            gate.check(
                (validation_dir / "checkpoint_best.pth.tar").is_file()
                and (validation_dir / "checkpoint_epoch_150.pth.tar").is_file(),
                "selected and epoch-150 artifacts are retained before latest is committed",
            )

            validation_best = legacy_train.load_tensor_checkpoint(
                validation_dir / "checkpoint_best.pth.tar", map_location="cpu")
            validation_milestone = legacy_train.load_tensor_checkpoint(
                validation_dir / "checkpoint_epoch_150.pth.tar", map_location="cpu")

            def copy_validation_namespace(name: str, *, best_state: Mapping[str, Any],
                                          milestone_state: Mapping[str, Any]) -> Path:
                destination = root / name
                destination.mkdir()
                spinr_train._atomic_torch_save(validation_latest, destination / "checkpoint_latest.pth.tar")
                spinr_train._atomic_torch_save(best_state, destination / "checkpoint_best.pth.tar")
                spinr_train._atomic_torch_save(
                    milestone_state, destination / "checkpoint_epoch_150.pth.tar")
                spinr_train._atomic_json_save(
                    {"history": validation_latest["history"]}, destination / "metrics_history.json")
                return destination

            changed_best_normalization = dict(validation_best)
            changed_best_normalization["normalization"] = {
                **validation_best["normalization"],
                "training_mean_raw_power": 2.0,
            }
            changed_best_dir = copy_validation_namespace(
                "changed-best-normalization", best_state=changed_best_normalization,
                milestone_state=validation_milestone)
            try:
                invoke(
                    "changed-best-normalization",
                    resume=str(changed_best_dir / "checkpoint_latest.pth.tar"),
                    interrupt_after_update=None, signal_during_validation=False, epochs=2)
            except ValueError:
                changed_best_normalization_rejected = True
            else:
                changed_best_normalization_rejected = False

            changed_milestone_normalization = dict(validation_milestone)
            changed_milestone_normalization["normalization"] = {
                **validation_milestone["normalization"],
                "initial_scale_training_ids": [1, 0],
            }
            changed_milestone_dir = copy_validation_namespace(
                "changed-milestone-normalization", best_state=validation_best,
                milestone_state=changed_milestone_normalization)
            try:
                invoke(
                    "changed-milestone-normalization",
                    resume=str(changed_milestone_dir / "checkpoint_latest.pth.tar"),
                    interrupt_after_update=None, signal_during_validation=False, epochs=2)
            except ValueError:
                changed_milestone_normalization_rejected = True
            else:
                changed_milestone_normalization_rejected = False
            gate.check(
                changed_best_normalization_rejected
                and changed_milestone_normalization_rejected
                and mode["materialize_calls"] == 0,
                "committed best/milestone normalization or fixed-scale IDs cannot diverge before raw materialization",
            )

            validation_resume_trace = invoke(
                "validation", resume=str(validation_dir / "checkpoint_latest.pth.tar"),
                interrupt_after_update=None,
                signal_during_validation=False,
                epochs=2,
            )
            validation_final = legacy_train.load_tensor_checkpoint(
                validation_dir / "checkpoint_final.pth.tar", map_location="cpu")
            gate.check(
                len(validation_trace) == 2 and len(validation_resume_trace) == 2
                and (validation_dir / "checkpoint_best.pth.tar").is_file()
                and (validation_dir / "checkpoint_epoch_150.pth.tar").is_file()
                and validation_final["execution"]["phase"] == "completed",
                "post-resume continuation retains selected/milestone artifacts and only advances the next epoch",
            )

            try:
                invoke(
                    "post-budget", resume=None, interrupt_after_update=None,
                    signal_during_validation=False, crash_after_committed_epoch=1)
            except SimulatedProcessLoss:
                post_budget_crash_observed = True
            else:
                post_budget_crash_observed = False
            post_budget_dir = root / "post-budget"
            post_budget_latest = legacy_train.load_tensor_checkpoint(
                post_budget_dir / "checkpoint_latest.pth.tar", map_location="cpu")
            post_budget_resume_trace = invoke(
                "post-budget", resume=str(post_budget_dir / "checkpoint_latest.pth.tar"),
                interrupt_after_update=None, signal_during_validation=False)
            post_budget_final = legacy_train.load_tensor_checkpoint(
                post_budget_dir / "checkpoint_final.pth.tar", map_location="cpu")
            gate.check(
                post_budget_crash_observed
                and post_budget_latest["epoch_index"] == 1
                and post_budget_latest["execution"]["phase"] == "updates"
                and not post_budget_resume_trace
                and post_budget_final["execution"]["stop_reason"] == "epoch_budget_reached",
                "hard loss after committed latest/history reaches the epoch terminal state without another update",
            )

            original_plateau = spinr_train.plateau_reached
            spinr_train.plateau_reached = lambda history: bool(history)
            try:
                try:
                    invoke(
                        "post-plateau", resume=None, interrupt_after_update=None,
                        signal_during_validation=False, crash_after_committed_epoch=1, epochs=2)
                except SimulatedProcessLoss:
                    post_plateau_crash_observed = True
                else:
                    post_plateau_crash_observed = False
                post_plateau_dir = root / "post-plateau"
                post_plateau_resume_trace = invoke(
                    "post-plateau", resume=str(post_plateau_dir / "checkpoint_latest.pth.tar"),
                    interrupt_after_update=None, signal_during_validation=False, epochs=2)
            finally:
                spinr_train.plateau_reached = original_plateau
            post_plateau_final = legacy_train.load_tensor_checkpoint(
                root / "post-plateau" / "checkpoint_final.pth.tar", map_location="cpu")
            gate.check(
                post_plateau_crash_observed
                and not post_plateau_resume_trace
                and post_plateau_final["execution"]["stop_reason"]
                    == "plateau_three_consecutive_10_epoch_windows",
                "hard loss after committed plateau history records terminal plateau policy before any next update",
            )

            cap_first_trace = invoke(
                "pending-budget-cap", resume=None, interrupt_after_update=None,
                signal_during_validation=True, epochs=2)
            cap_dir = root / "pending-budget-cap"
            try:
                invoke(
                    "pending-budget-cap", resume=str(cap_dir / "checkpoint_latest.pth.tar"),
                    interrupt_after_update=None, signal_during_validation=False,
                    crash_after_pending_epoch=1, epochs=2)
            except SimulatedProcessLoss:
                cap_pending_crash_observed = True
            else:
                cap_pending_crash_observed = False
            try:
                invoke(
                    "pending-budget-cap", resume=str(cap_dir / "checkpoint_latest.pth.tar"),
                    interrupt_after_update=None, signal_during_validation=False, epochs=1)
            except ValueError:
                cap_lower_budget_rejected = True
            else:
                cap_lower_budget_rejected = False
            gate.check(
                len(cap_first_trace) == 2
                and cap_pending_crash_observed
                and cap_lower_budget_rejected
                and mode["materialize_calls"] == 0,
                "pending epoch that would cross a lowered resume budget is rejected before raw materialization",
            )

            partial_first_trace = invoke(
                "partial-budget-cap", resume=None, interrupt_after_update=None,
                signal_during_validation=True, epochs=2)
            partial_cap_dir = root / "partial-budget-cap"
            partial_second_trace = invoke(
                "partial-budget-cap",
                resume=str(partial_cap_dir / "checkpoint_latest.pth.tar"),
                interrupt_after_update=1, signal_during_validation=False, epochs=2)
            partial_cap_latest = legacy_train.load_tensor_checkpoint(
                partial_cap_dir / "checkpoint_latest.pth.tar", map_location="cpu")
            try:
                invoke(
                    "partial-budget-cap",
                    resume=str(partial_cap_dir / "checkpoint_latest.pth.tar"),
                    interrupt_after_update=None, signal_during_validation=False, epochs=1)
            except ValueError:
                partial_lower_budget_rejected = True
            else:
                partial_lower_budget_rejected = False
            gate.check(
                len(partial_first_trace) == 2
                and len(partial_second_trace) == 1
                and partial_cap_latest["execution"]["phase"] == "updates"
                and partial_cap_latest["execution"]["partial_epoch"]["completed_updates"] == 1
                and partial_lower_budget_rejected
                and mode["materialize_calls"] == 0,
                "nonzero-cursor epoch that would cross a lowered resume budget is rejected before raw materialization",
            )

            original_plateau = spinr_train.plateau_reached
            spinr_train.plateau_reached = lambda history: bool(history)
            try:
                plateau_trace = invoke(
                    "plateau", resume=None, interrupt_after_update=2,
                    signal_during_validation=True, epochs=2)
            finally:
                spinr_train.plateau_reached = original_plateau
            plateau_final = legacy_train.load_tensor_checkpoint(
                root / "plateau" / "checkpoint_final.pth.tar", map_location="cpu")
            gate.check(
                plateau_trace == baseline_trace
                and plateau_final["execution"]["stop_reason"] == "plateau_three_consecutive_10_epoch_windows",
                "TERM at a qualified plateau persists terminal policy before any next epoch can begin",
            )

            budget_trace = invoke(
                "budget", resume=None, interrupt_after_update=2, signal_during_validation=True)
            budget_final = legacy_train.load_tensor_checkpoint(
                root / "budget" / "checkpoint_final.pth.tar", map_location="cpu")
            gate.check(
                budget_trace == baseline_trace
                and budget_final["execution"]["phase"] == "completed"
                and budget_final["execution"]["stop_reason"] == "epoch_budget_reached",
                "TERM at the epoch budget records a durable terminal stop rather than requiring another update",
            )
    finally:
        for name, value in original_values.items():
            setattr(spinr_train, name, value)


def stage_allocated_pace(gate: Gate, args: argparse.Namespace) -> None:
    if not args.allocated_pace:
        return
    print("Stage H: allocated-node real B787 authorized-only smoke", flush=True)
    if args.pace_npz is None or args.role_manifest is None:
        raise ValueError("--allocated-pace requires --pace-npz and --role-manifest")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("--allocated-pace requires a CUDA allocation; it must not silently fall back to CPU")
    device = torch.device("cuda")
    spinr_train._disable_tf32()
    views, contract = spinr_train.load_b787_development_views(
        npz_path=args.pace_npz,
        manifest_path=args.role_manifest,
        resume_checkpoint=None,
    )
    gate.check(contract["response_access"]["reserved_test_materialized"] is False
               and contract["response_access"]["unused_materialized"] is False,
               "real B787 adapter retains sealed test/unused access denial")
    gate.check(views.materialized_response_bytes > 4 * 1024 ** 3,
               "real adapter reports its intentional authorized host cache for RSS gating")
    memory_report = spinr_train._enforce_memory_gates(device=device, host_rss_limit_gib=args.host_rss_limit_gib)
    print("  ingest memory: " + json.dumps(memory_report, sort_keys=True), flush=True)
    torch.manual_seed(42)
    model = SpinrStyleINR().to(device)
    points, volume = midpoint_grid(48, device=device, dtype=torch.float64)
    frequencies = torch.as_tensor(views.frequencies_hz, device=device, dtype=torch.float64)
    kvector = get_kvector(frequencies, cc).to(dtype=torch.float64)
    training_power = views.raw_training_mean_power()
    a_init, init_ids, _obs_energy, _pred_energy = spinr_train.estimate_initial_output_scale(
        model=model, views=views, points_m=points, cell_volume_m3=volume,
        frequencies_hz=frequencies, kvector=kvector, neural_point_tile=4096,
        renderer_point_tile=65536, pair_tile=16, device=device,
    )
    optimizer = Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
    loss, grad_norm = spinr_train.logical_batch_update(
        model=model, optimizer=optimizer, source_ids=views.role_ids("train")[:4], views=views,
        points_m=points, cell_volume_m3=volume, initial_output_scale=a_init,
        frequencies_hz=frequencies, kvector=kvector, training_mean_raw_power=training_power,
        neural_point_tile=4096, renderer_point_tile=65536, pair_tile=16, device=device,
    )
    gate.check(math.isfinite(loss) and math.isfinite(grad_norm), "real B787 all-bin/all-pair B=4 update is finite")
    gate.check(init_ids == views.role_ids("train")[:32], "real B787 initial scale uses exactly first 32 canonical train IDs")
    report = spinr_train._enforce_memory_gates(device=device, host_rss_limit_gib=args.host_rss_limit_gib)
    print("  full-update memory: " + json.dumps(report, sort_keys=True), flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allocated-pace", action="store_true")
    parser.add_argument("--pace-npz", default=None)
    parser.add_argument("--role-manifest", default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    parser.add_argument("--host-rss-limit-gib", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.allocated_pace:
        if args.host_rss_limit_gib is None:
            raise ValueError("--allocated-pace requires --host-rss-limit-gib from the allocated Slurm request")
    elif args.pace_npz is not None or args.role_manifest is not None:
        raise ValueError("PACE paths require --allocated-pace; default validation is data-free")
    gate = Gate()
    stage_recipe_and_encoding(gate)
    stage_frequency_fft_and_axis(gate)
    stage_sealed_header_preflight(gate)
    stage_raw_complex_adapter(gate)
    stage_product_renderer_and_real_vjp(gate)
    stage_preflight_namespace_and_acquisition(gate)
    stage_tiled_b4_gradient_and_resume(gate)
    stage_actual_run_resume_state_machine(gate)
    stage_allocated_pace(gate, args)
    print(f"SpINR-style validation passed: {gate.count} checks.", flush=True)


if __name__ == "__main__":
    main()
