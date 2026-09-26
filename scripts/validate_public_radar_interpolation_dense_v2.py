#!/usr/bin/env python
"""CPU contract gates for the fixed-planar serialized PublicRadar v2 path."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_local(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PLANAR = load_local(
    "rift_fixed_planar_scene_v2_validation",
    PROJECT_ROOT / "rift" / "fixed_planar_scene.py",
)
SERIAL = load_local(
    "rift_serialized_range_operator_v2_validation",
    PROJECT_ROOT / "rift" / "serialized_range_operator.py",
)
GOTCHA_DOMAIN = load_local(
    "rift_gotcha_full_domain_validation",
    PROJECT_ROOT / "rift" / "gotcha_domain.py",
)

from rift.coherent_radar_geometry import (
    MONOSTATIC_FAR_FIELD_REFERENCE,
    MONOSTATIC_NEAR_FIELD_REFERENCE,
)
from rift.config import cc
from rift.forward_operator import get_kvector
from rift.range_operator import range_adjoint_operator, range_forward_operator


def gate(name, passed):
    if not bool(passed):
        raise AssertionError(name)
    print(f"PASS: {name}", flush=True)


def relative_l2(actual, expected):
    return float(
        torch.linalg.vector_norm(actual - expected)
        / torch.linalg.vector_norm(expected).clamp_min(1.0e-30)
    )


def coordinate_gate():
    model = PLANAR.FixedPlanarSHScene(5, 4, 2.0, 0, torch.device("cpu"))
    actual = torch.cat(list(model.position_chunks(3)), dim=0)
    x = torch.linspace(-1.6, 1.6, 5)
    y = torch.linspace(-1.5, 1.5, 4)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    expected = torch.stack((xx, yy, torch.zeros_like(xx)), dim=-1).reshape(-1, 3)
    coordinate_error = float((actual - expected).abs().max().item())
    coordinate_atol = 8.0 * torch.finfo(actual.dtype).eps * max(1.0, model.extent)
    print(
        f"INFO: planar coordinate max_abs_error={coordinate_error:.9g} "
        f"atol={coordinate_atol:.9g} dtype={actual.dtype}",
        flush=True,
    )
    gate(
        "planar cell-centre coordinates and flatten order",
        actual.shape == expected.shape
        and torch.allclose(actual, expected, rtol=0.0, atol=coordinate_atol),
    )
    gate("planar representation has no adaptive methods", not any(
        hasattr(model, name) for name in ("prune", "grow", "split")
    ))


def operator_gate(degree, propagation_model, geometry_label):
    torch.manual_seed(100 + degree)
    model = PLANAR.FixedPlanarSHScene(
        5, 4, 0.4, degree, torch.device("cpu"), init_scale=0.02
    )
    dtheta = torch.tensor([[0.4]], dtype=torch.float32)
    dphi = torch.tensor([[1.1]], dtype=torch.float32)
    positions = torch.cat(list(model.position_chunks(3)), dim=0)
    weights = torch.cat(
        [value for _, value in model.scatterer_chunks(dtheta, dphi, 3)], dim=0
    )
    frequencies = torch.linspace(9.0e9, 9.2e9, 17, dtype=torch.float64)
    kvector = get_kvector(frequencies, cc)
    platform = torch.tensor([[7.0, 4.0, 3.0]], dtype=torch.float64)
    kwargs = {
        "phase_sign": -1.0,
        "freq_indices": torch.arange(frequencies.numel()),
        "pair_chunk": 1,
        "compute_dtype": torch.float64,
        "range_model": "none",
        "propagation_model": propagation_model,
        "reference_range_m": 7.5,
        "scene_center_m": (0.0, 0.0, 0.0),
    }
    reference = range_forward_operator(
        frequencies,
        kvector,
        platform,
        platform,
        positions,
        weights,
        point_chunk=3,
        **kwargs,
    )
    serialized = SERIAL.range_forward_operator_chunks(
        frequencies,
        kvector,
        platform,
        platform,
        lambda: model.scatterer_chunks(dtheta, dphi, 3),
        **kwargs,
    )
    gate(
        f"degree-{degree} {geometry_label} serialized forward equality",
        torch.allclose(serialized, reference, rtol=1.0e-12, atol=1.0e-12),
    )

    model.zero_grad(set_to_none=True)
    reference_loss = reference.abs().square().sum()
    reference_loss.backward()
    reference_re = model.w_re.grad.detach().clone()
    reference_im = model.w_im.grad.detach().clone()
    model.zero_grad(set_to_none=True)
    serialized = SERIAL.range_forward_operator_chunks(
        frequencies,
        kvector,
        platform,
        platform,
        lambda: model.scatterer_chunks(dtheta, dphi, 3),
        **kwargs,
    )
    serialized.abs().square().sum().backward()
    gate(
        f"degree-{degree} {geometry_label} serialized real gradient equality",
        torch.allclose(model.w_re.grad, reference_re, rtol=1.0e-10, atol=1.0e-10),
    )
    gate(
        f"degree-{degree} {geometry_label} serialized imaginary gradient equality",
        torch.allclose(model.w_im.grad, reference_im, rtol=1.0e-10, atol=1.0e-10),
    )

    residual = torch.complex(
        torch.randn_like(reference.real), torch.randn_like(reference.real)
    )
    reference_adjoint = range_adjoint_operator(
        frequencies,
        kvector,
        platform,
        platform,
        positions,
        residual,
        point_chunk=3,
        **kwargs,
    )
    serialized_adjoint = SERIAL.range_adjoint_operator_chunks(
        frequencies,
        kvector,
        platform,
        platform,
        lambda: model.position_chunks(3),
        residual,
        **kwargs,
    )
    gate(
        f"degree-{degree} {geometry_label} serialized adjoint equality",
        torch.allclose(
            serialized_adjoint, reference_adjoint, rtol=1.0e-12, atol=1.0e-12
        ),
    )


def masked_operator_gate():
    """The optional spatial mask must define an exact restricted operator."""
    torch.manual_seed(911)
    model = PLANAR.FixedPlanarSHScene(
        5, 4, 0.4, 0, torch.device("cpu"), init_scale=0.02
    )
    dtheta = torch.tensor([[0.4]], dtype=torch.float32)
    dphi = torch.tensor([[1.1]], dtype=torch.float32)
    positions = torch.cat(list(model.position_chunks(20)), dim=0)
    weights = torch.cat(
        [value for _, value in model.scatterer_chunks(dtheta, dphi, 20)], dim=0
    )
    keep = positions[:, 0] >= 0.0
    point_mask = lambda chunk: chunk[:, 0] >= 0.0
    frequencies = torch.linspace(9.0e9, 9.2e9, 17, dtype=torch.float64)
    kvector = get_kvector(frequencies, cc)
    platform = torch.tensor([[7.0, 4.0, 3.0]], dtype=torch.float64)
    kwargs = {
        "phase_sign": -1.0,
        "freq_indices": torch.arange(frequencies.numel()),
        "pair_chunk": 1,
        "compute_dtype": torch.float64,
        "range_model": "none",
        "propagation_model": MONOSTATIC_NEAR_FIELD_REFERENCE,
        "reference_range_m": 7.5,
        "scene_center_m": (0.0, 0.0, 0.0),
    }
    reference = range_forward_operator(
        frequencies,
        kvector,
        platform,
        platform,
        positions[keep],
        weights[keep],
        point_chunk=3,
        **kwargs,
    )

    def scatterer_chunks(chunk_size):
        return lambda: model.scatterer_chunks(dtheta, dphi, chunk_size)

    masked_small = SERIAL.range_forward_operator_chunks(
        frequencies,
        kvector,
        platform,
        platform,
        scatterer_chunks(3),
        point_mask=point_mask,
        **kwargs,
    )
    masked_large = SERIAL.range_forward_operator_chunks(
        frequencies,
        kvector,
        platform,
        platform,
        scatterer_chunks(7),
        point_mask=point_mask,
        **kwargs,
    )
    gate(
        "masked serialized forward equality",
        torch.allclose(masked_small, reference, rtol=1.0e-12, atol=1.0e-12),
    )
    gate(
        "masked serialized forward chunk invariance",
        torch.allclose(masked_small, masked_large, rtol=1.0e-12, atol=1.0e-12),
    )

    model.zero_grad(set_to_none=True)
    masked_small.abs().square().sum().backward()
    gate(
        "masked serialized invalid real gradients are zero",
        bool((model.w_re.grad[~keep] == 0).all()),
    )
    gate(
        "masked serialized invalid imaginary gradients are zero",
        bool((model.w_im.grad[~keep] == 0).all()),
    )

    residual = torch.complex(
        torch.randn_like(reference.real), torch.randn_like(reference.real)
    )
    reference_valid = range_adjoint_operator(
        frequencies,
        kvector,
        platform,
        platform,
        positions[keep],
        residual,
        point_chunk=3,
        **kwargs,
    )
    expected_adjoint = torch.zeros(
        positions.shape[0], dtype=reference_valid.dtype
    ).index_copy(0, torch.nonzero(keep, as_tuple=False).squeeze(-1), reference_valid)
    masked_adjoint = SERIAL.range_adjoint_operator_chunks(
        frequencies,
        kvector,
        platform,
        platform,
        lambda: model.position_chunks(3),
        residual,
        point_mask=point_mask,
        **kwargs,
    )
    gate(
        "masked serialized adjoint equality",
        torch.allclose(masked_adjoint, expected_adjoint, rtol=1.0e-12, atol=1.0e-12),
    )
    lhs = (masked_small.conj() * residual).sum()
    rhs = (weights.conj() * masked_adjoint).sum()
    gate(
        "masked serialized forward-adjoint inner product",
        torch.allclose(lhs, rhs, rtol=1.0e-10, atol=1.0e-10),
    )


def aligned_view_operator_gate(degree, propagation_model, geometry_label, batch_size):
    """An aligned view block must equal independent scalar renderings."""
    torch.manual_seed(1700 + 10 * degree + batch_size)
    model = PLANAR.FixedPlanarSHScene(
        5, 4, 0.4, degree, torch.device("cpu"), init_scale=0.02
    )
    dtheta = torch.linspace(0.25, 0.75, batch_size, dtype=torch.float32)
    dphi = torch.linspace(0.4, 1.6, batch_size, dtype=torch.float32)
    offsets = torch.linspace(-0.3, 0.3, batch_size, dtype=torch.float64)
    platforms = torch.stack(
        (
            7.0 + offsets,
            4.0 - 0.5 * offsets,
            3.0 + 0.25 * offsets,
        ),
        dim=-1,
    )
    frequencies = torch.linspace(9.0e9, 9.2e9, 17, dtype=torch.float64)
    kvector = get_kvector(frequencies, cc)
    common = {
        "phase_sign": -1.0,
        "freq_indices": torch.arange(frequencies.numel()),
        "compute_dtype": torch.float64,
        "range_model": "none",
        "propagation_model": propagation_model,
        "reference_range_m": 7.5,
        "scene_center_m": (0.0, 0.0, 0.0),
    }

    # The legacy scalar path uses an FP32 matrix-vector reduction for every
    # view, while the batched path deliberately uses one FP32 matrix-matrix
    # multiply.  Those two valid reduction orders are not bit-identical for
    # degree > 0.  Check that compatibility separately at an FP32-scaled
    # tolerance, then source the operator reference from the exact same
    # batched weights so the range-kernel gate is not confounded by GEMV/GEMM
    # rounding.
    with torch.no_grad():
        legacy_weights = torch.stack(
            [
                torch.cat(
                    [
                        weights
                        for _, weights in model.scatterer_chunks(
                            dtheta[index].reshape(1, 1),
                            dphi[index].reshape(1, 1),
                            3,
                        )
                    ]
                )
                for index in range(batch_size)
            ],
            dim=1,
        )
        batched_weights = torch.cat(
            [weights for _, weights in model.scatterer_view_chunks(dtheta, dphi, 3)],
            dim=0,
        )
        other_chunk_weights = torch.cat(
            [weights for _, weights in model.scatterer_view_chunks(dtheta, dphi, 7)],
            dim=0,
        )
        weight_relative_error = relative_l2(batched_weights, legacy_weights)
        weight_max_abs_error = float((batched_weights - legacy_weights).abs().max())
        weight_chunk_relative_error = relative_l2(
            other_chunk_weights, batched_weights
        )
    weight_label = f"degree-{degree} aligned B={batch_size} SH weights"
    print(
        f"INFO: {weight_label} relative_error={weight_relative_error:.9g} "
        f"max_abs_error={weight_max_abs_error:.9g}",
        flush=True,
    )
    gate(f"{weight_label} FP32 compatibility", weight_relative_error <= 5.0e-6)
    print(
        f"INFO: {weight_label} chunk-size relative_error="
        f"{weight_chunk_relative_error:.9g}",
        flush=True,
    )
    gate(
        f"{weight_label} FP32 point-chunk compatibility",
        weight_chunk_relative_error <= 5.0e-6,
    )

    def aligned_column_chunks(view_index, chunk_size):
        def factory():
            for positions, weights in model.scatterer_view_chunks(
                dtheta, dphi, chunk_size
            ):
                yield positions, weights[:, view_index]

        return factory

    def scalar_predictions():
        values = []
        for view_index in range(batch_size):
            platform = platforms[view_index].reshape(1, 3)
            value = SERIAL.range_forward_operator_chunks(
                frequencies,
                kvector,
                platform,
                platform,
                aligned_column_chunks(view_index, 3),
                pair_chunk=1,
                **common,
            )
            values.append(value[:, 0, 0])
        return torch.stack(values)

    reference = scalar_predictions()
    batched = SERIAL.range_forward_operator_aligned_view_chunks(
        frequencies,
        kvector,
        platforms,
        platforms,
        lambda: model.scatterer_view_chunks(dtheta, dphi, 3),
        **common,
    )
    label = f"degree-{degree} {geometry_label} aligned B={batch_size}"
    forward_relative_error = relative_l2(batched, reference)
    forward_max_abs_error = float((batched - reference).abs().max())
    print(
        f"INFO: {label} forward relative_error={forward_relative_error:.9g} "
        f"max_abs_error={forward_max_abs_error:.9g}",
        flush=True,
    )
    gate(
        f"{label} forward equality",
        batched.shape == reference.shape
        and torch.allclose(batched, reference, rtol=1.0e-11, atol=1.0e-11),
    )
    batched_recomputed_other_chunk = SERIAL.range_forward_operator_aligned_view_chunks(
        frequencies,
        kvector,
        platforms,
        platforms,
        lambda: model.scatterer_view_chunks(dtheta, dphi, 7),
        **common,
    )
    recomputed_chunk_relative_error = relative_l2(
        batched_recomputed_other_chunk, batched
    )
    print(
        f"INFO: {label} recomputed point-chunk relative_error="
        f"{recomputed_chunk_relative_error:.9g}",
        flush=True,
    )
    gate(
        f"{label} production FP32 point-chunk compatibility",
        recomputed_chunk_relative_error <= 5.0e-6,
    )

    positions = torch.cat(list(model.position_chunks(3)), dim=0)

    def fixed_weight_chunks(chunk_size):
        def factory():
            for start in range(0, model.n_points, chunk_size):
                stop = min(start + chunk_size, model.n_points)
                yield positions[start:stop], batched_weights[start:stop]

        return factory

    fixed_small = SERIAL.range_forward_operator_aligned_view_chunks(
        frequencies,
        kvector,
        platforms,
        platforms,
        fixed_weight_chunks(3),
        **common,
    )
    fixed_large = SERIAL.range_forward_operator_aligned_view_chunks(
        frequencies,
        kvector,
        platforms,
        platforms,
        fixed_weight_chunks(7),
        **common,
    )
    gate(
        f"{label} range-kernel point-chunk invariance",
        torch.allclose(fixed_large, fixed_small, rtol=1.0e-11, atol=1.0e-11),
    )

    target = torch.complex(
        torch.randn_like(reference.real), torch.randn_like(reference.real)
    )
    model.zero_grad(set_to_none=True)
    (reference - target).abs().square().sum().backward()
    reference_re = model.w_re.grad.detach().clone()
    reference_im = model.w_im.grad.detach().clone()
    model.zero_grad(set_to_none=True)
    batched = SERIAL.range_forward_operator_aligned_view_chunks(
        frequencies,
        kvector,
        platforms,
        platforms,
        lambda: model.scatterer_view_chunks(dtheta, dphi, 3),
        **common,
    )
    (batched - target).abs().square().sum().backward()
    for component, actual, expected in (
        ("real", model.w_re.grad, reference_re),
        ("imaginary", model.w_im.grad, reference_im),
    ):
        relative_error = float(
            torch.linalg.vector_norm(actual - expected)
            / torch.linalg.vector_norm(expected).clamp_min(1.0e-30)
        )
        print(
            f"INFO: {label} {component} gradient relative_error={relative_error:.9g}",
            flush=True,
        )
        gate(f"{label} {component} gradient equality", relative_error <= 1.0e-7)


def aligned_view_shape_gate():
    model = PLANAR.FixedPlanarSHScene(
        3, 2, 0.2, 0, torch.device("cpu"), init_scale=0.01
    )
    frequencies = torch.linspace(9.0e9, 9.1e9, 9, dtype=torch.float64)
    kvector = get_kvector(frequencies, cc)
    platform = torch.tensor([[7.0, 4.0, 3.0]], dtype=torch.float64)
    kwargs = {
        "phase_sign": -1.0,
        "compute_dtype": torch.float64,
        "range_model": "none",
        "propagation_model": MONOSTATIC_NEAR_FIELD_REFERENCE,
        "reference_range_m": 7.5,
    }
    try:
        SERIAL.range_forward_operator_aligned_view_chunks(
            frequencies,
            kvector,
            platform.unsqueeze(1),
            platform.unsqueeze(1),
            lambda: model.scatterer_view_chunks(
                torch.tensor([0.4]), torch.tensor([1.1]), 3
            ),
            **kwargs,
        )
    except ValueError:
        gate("aligned view operator rejects a latent antenna axis", True)
    else:
        gate("aligned view operator rejects a latent antenna axis", False)


def gotcha_full_domain_dataset_gate(npz_path):
    """Validate the frozen 100 m square against the real converted geometry."""
    npz_path = Path(npz_path).resolve()
    with np.load(npz_path, allow_pickle=True) as archive:
        required = {"metadata_json", "frequencies_hz", "viewpoint_positions"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"GOTCHA dataset lacks full-domain fields: {missing}")
        raw_metadata = np.asarray(archive["metadata_json"])
        if raw_metadata.size != 1:
            raise ValueError("GOTCHA metadata_json must contain exactly one value")
        metadata = json.loads(str(raw_metadata.item()))
        frequencies = np.asarray(archive["frequencies_hz"], dtype=np.float64)
        viewpoints = np.asarray(archive["viewpoint_positions"], dtype=np.float64)

    gate(
        "GOTCHA full-domain dataset identity",
        metadata.get("dataset") == "gotcha"
        and metadata.get("pass_id") == "pass2"
        and metadata.get("polarization") == "hh",
    )
    gate(
        "GOTCHA full-domain scene centre",
        tuple(float(value) for value in metadata.get("scene_center_m", ()))
        == (0.0, 0.0, 0.0),
    )
    frequency_grid = GOTCHA_DOMAIN.validate_gotcha_frequency_grid(
        frequencies,
        nominal_spacing_hz=float(metadata["frequency_spacing_hz"]),
        unambiguous_range_m=float(metadata["unambiguous_range_m"]),
    )
    gate("GOTCHA nominal frequency grid with bounded source quantization", True)
    gate("GOTCHA frequency grid matches range-operator endpoint model", True)
    reference_range_m = float(metadata["reference_range_m"])
    summary = GOTCHA_DOMAIN.validate_gotcha_planar_square_views(
        viewpoints,
        reference_range_m=reference_range_m,
        frequency_step_hz=frequency_grid["conservative_spacing_hz"],
        scene_center_m=metadata["scene_center_m"],
    )
    gate(
        "GOTCHA complete z=0 100 m square is nonwrapping for every view",
        summary["view_count"] == viewpoints.shape[0]
        and summary["minimum_lower_window_margin_m"] > 0.0
        and summary["minimum_upper_window_margin_m"] > 0.0,
    )
    corners = torch.tensor(
        [
            [-50.0, -50.0, 0.0],
            [-50.0, 50.0, 0.0],
            [50.0, -50.0, 0.0],
            [50.0, 50.0, 0.0],
        ],
        dtype=torch.float64,
    )
    corner_mask = GOTCHA_DOMAIN.gotcha_nonwrapping_chunk_mask(
        corners,
        torch.as_tensor(viewpoints, dtype=torch.float64),
        reference_range_m=reference_range_m,
        frequency_step_hz=frequency_grid["conservative_spacing_hz"],
        scene_center_m=metadata["scene_center_m"],
    )
    gate("GOTCHA full-domain corner mask is the identity", bool(corner_mask.all()))
    nx = ny = 1776
    pitch = 100.0 / nx
    gate(
        "GOTCHA full-domain pitch preserves the dense-v2 resolution",
        math.isclose(pitch, 0.05630630630630631, rel_tol=0.0, abs_tol=1.0e-15)
        and nx * ny == 3_154_176,
    )
    print(
        "INFO: GOTCHA full-domain preflight "
        f"views={viewpoints.shape[0]} points={nx * ny} pitch_m={pitch:.12f} "
        f"nominal_df_hz={frequency_grid['nominal_spacing_hz']:.3f} "
        f"max_df_hz={frequency_grid['max_spacing_hz']:.3f} "
        f"endpoint_df_hz={frequency_grid['endpoint_affine_spacing_hz']:.6f} "
        f"endpoint_residual_ratio="
        f"{frequency_grid['endpoint_affine_residual_fraction']:.9g} "
        f"source_quantization_bound_hz="
        f"{frequency_grid['float32_source_step_tolerance_hz']:.3f} "
        f"lower_margin_m={summary['minimum_lower_window_margin_m']:.9f} "
        f"upper_margin_m={summary['minimum_upper_window_margin_m']:.9f}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gotcha-npz")
    args = parser.parse_args()
    coordinate_gate()
    operator_gate(0, MONOSTATIC_FAR_FIELD_REFERENCE, "far-field-reference")
    operator_gate(3, MONOSTATIC_FAR_FIELD_REFERENCE, "far-field-reference")
    operator_gate(3, MONOSTATIC_NEAR_FIELD_REFERENCE, "near-field-reference")
    masked_operator_gate()
    for batch_size in (1, 2, 4, 8):
        aligned_view_operator_gate(
            0,
            MONOSTATIC_NEAR_FIELD_REFERENCE,
            "near-field-reference",
            batch_size,
        )
        aligned_view_operator_gate(
            3,
            MONOSTATIC_FAR_FIELD_REFERENCE,
            "far-field-reference",
            batch_size,
        )
        aligned_view_operator_gate(
            3,
            MONOSTATIC_NEAR_FIELD_REFERENCE,
            "near-field-reference",
            batch_size,
        )
    aligned_view_shape_gate()
    if args.gotcha_npz:
        gotcha_full_domain_dataset_gate(args.gotcha_npz)
    print("All fixed-planar serialized v2 CPU gates passed.", flush=True)


if __name__ == "__main__":
    main()
