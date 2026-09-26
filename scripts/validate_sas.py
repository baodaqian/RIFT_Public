#!/usr/bin/env python
"""CPU contract gates for the shared SAS cache, renderer, and both fields."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rift.rift_sas import AdaptiveRIFTSASField, ComplexSHSonarField, RIFTSASGrid
from rift.airsas_contract import (
    EXPECTED_CROP_BINS,
    EXPECTED_GEOMETRY_SHAPE,
    EXPECTED_TX_SHAPE,
    validate_system_data_5k,
)
from rift.sas_dataset import restricted_load_system_data, schema_dict
from rift.sas_operator import ellipsoid_samples, normalize, render_sas_bins, two_way_transmittance
from rift.sh_sas import SHSASField, real_sh_basis_for_directions
from rift.sparse_scene import AdaptivePointSHScene


def check(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)
    print(f"PASS {label}")


def build_adapter(base, degree: int) -> ComplexSHSonarField:
    return ComplexSHSonarField(
        base,
        torch.tensor([-0.2, -0.2, 0.0]),
        torch.tensor([0.2, 0.2, 0.3]),
        degree,
        query_chunk=4096,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system-data", required=True)
    parser.add_argument("--reed-root", default="external/Reed_SAS_reference")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.reed_root).resolve()))

    system = restricted_load_system_data(args.system_data)
    try:
        contract = validate_system_data_5k(args.system_data, system)
    except ValueError as exc:
        raise AssertionError(str(exc)) from exc
    geometry = schema_dict(system["geometry"])
    crop = schema_dict(system["crop_settings"])
    check(tuple(contract["tx_shape"]) == EXPECTED_TX_SHAPE, "safe schema Armadillo5k Tx/Rx shape")
    check(contract["crop_bins"] == EXPECTED_CROP_BINS, "safe schema Armadillo5k radial bins")
    check(tuple(contract["geometry_grid_shape"]) == EXPECTED_GEOMETRY_SHAPE, "safe schema Armadillo5k geometry")

    device = torch.device("cpu")
    corners = torch.as_tensor(geometry["corners"], dtype=torch.float32)
    tx = torch.as_tensor(system["tx_coords"][0], dtype=torch.float32)
    rx = torch.as_tensor(system["rx_coords"][0], dtype=torch.float32)
    radii = torch.linspace(float(crop["min_dist"]), float(crop["max_dist"]), 4)
    points, directions = ellipsoid_samples(
        radii, tx, rx, corners, num_rays=64, point_at_center=True, transmit_from_tx=True
    )
    check(points.ndim == 3 and points.shape[0] == 4 and points.shape[-1] == 3, "ellipsoid shape")
    check(torch.isfinite(points).all().item(), "ellipsoid finite")
    check(torch.allclose(torch.linalg.vector_norm(directions, dim=-1), torch.ones(directions.shape[0]), atol=1e-5), "ray unit norm")
    distance_sum = torch.linalg.vector_norm(points - tx, dim=-1) + torch.linalg.vector_norm(points - rx, dim=-1)
    check(torch.allclose(distance_sum, radii[:, None], rtol=2e-4, atol=2e-4), "bistatic ellipsoid invariant")

    density = torch.ones(4, 3)
    transmission = two_way_transmittance(radii, density, 2.0)
    check(torch.allclose(transmission[0], torch.ones(3)), "exclusive transmittance starts at one")
    check(bool((transmission[1:] <= transmission[:-1]).all()), "transmittance monotone")

    rift_grid = RIFTSASGrid(4, 1.0, device, max_degree=1, init_degree=1, init_scale=0.0)
    with torch.no_grad():
        for ix in range(4):
            rift_grid.w_re[ix, ..., 0] = float(ix)
        rift_grid.w_im.zero_()
    nodes = rift_grid.grid_positions.reshape(-1, 3)
    queried = rift_grid.query_coefficients(nodes)
    check(torch.allclose(queried[:, 0].real, rift_grid.w_re[..., 0].reshape(-1), atol=1e-6), "trilinear node exactness")
    center = rift_grid.query_coefficients(torch.zeros(1, 3))[0, 0].real
    check(abs(float(center) - 1.5) < 1e-6, "trilinear midpoint")

    fields = {
        "rift_sas": build_adapter(RIFTSASGrid(6, 1.0, device, max_degree=1, init_degree=1, init_scale=1e-2), 1),
        "sh_sas": build_adapter(
            SHSASField(
                extent=1.0,
                granularity=6,
                sh_degree=1,
                hidden_dim=8,
                hash_levels=2,
                hash_features=2,
                hash_base_resolution=2,
                hash_final_resolution=4,
                hash_log2_size=8,
                device=device,
            ),
            1,
        ),
    }
    for name, field in fields.items():
        predicted, aux = render_sas_bins(
            field,
            radii[:2],
            tx,
            rx,
            corners,
            num_rays=16,
            opacity_scale=1.0,
            normal_step=0.01,
        )
        loss = predicted.abs().square().mean()
        loss.backward()
        gradients = [p.grad for p in field.parameters() if p.requires_grad]
        check(predicted.shape == (2,) and torch.isfinite(predicted).all().item(), f"{name} forward")
        check(any(g is not None and torch.isfinite(g).all() for g in gradients), f"{name} backward")
        check(int(aux["actual_rays"]) > 0, f"{name} nonempty ray integral")

    # The receiver-to-point SH direction is explicit and differs from the
    # Tx-origin sampling direction for this separated bistatic pair.
    check(
        torch.allclose(
            aux["sh_directions"],
            normalize(aux["points"] - rx.reshape(1, 1, 3)),
            atol=1e-6,
        ),
        "receiver-to-point SH direction",
    )
    direction = torch.tensor([[0.3, 0.2, 1.0]], dtype=torch.float32)
    reversed_direction = -direction
    basis = real_sh_basis_for_directions(direction, 1)
    reversed_basis = real_sh_basis_for_directions(reversed_direction, 1)
    check(not torch.allclose(basis[:, 1:], reversed_basis[:, 1:]), "odd SH reversal is visible")

    # Output-bin selection must be a view of the same full-context render.
    grid = RIFTSASGrid(6, 1.0, device, max_degree=1, init_degree=1, init_scale=1e-2)
    with torch.no_grad():
        grid.w_re[..., 0] = 0.5
        grid.w_im[..., 0] = 0.0
    grid_field = build_adapter(grid, 1)
    subset_radii = torch.linspace(float(crop["min_dist"]), float(crop["max_dist"]), 4)
    full, full_aux = render_sas_bins(
        grid_field, subset_radii, tx, rx, corners,
        num_rays=16, opacity_scale=0.75, lambertian_ratio=1.0, normal_step=0.01,
    )
    subset, subset_aux = render_sas_bins(
        grid_field, subset_radii, tx, rx, corners,
        num_rays=16, opacity_scale=0.75, lambertian_ratio=1.0, normal_step=0.01,
        output_bin_indices=torch.tensor([1, 3]),
    )
    check(torch.all(full_aux["transmittance"] > 0).item(), "positive-opacity transmittance remains positive")
    check(full.abs().sum() > 0, "positive-opacity render is nonzero")
    check(torch.allclose(subset, full[[1, 3]], atol=1e-6, rtol=1e-5), "positive-opacity subset equals full render")
    check(torch.equal(subset_aux["output_bin_indices"], torch.tensor([1, 3])), "subset records output bins")

    # Adaptive coefficient/position differentiation and the in-place heir
    # contract are checked on a tiny CPU scene.
    adaptive_scene = AdaptivePointSHScene.from_regular_grid(
        2, 1.0, device, max_degree=1, init_degree=0, init_scale=0.0,
        capacity=64, compact_sh_eval=True,
    )
    with torch.no_grad():
        adaptive_scene.delta_raw[0] = torch.tensor([0.23, -0.17, 0.11])
        adaptive_scene.w_re[0, 0] = 0.7
        adaptive_scene.w_im[0, 0] = -0.2
    adaptive_field = build_adapter(
        AdaptiveRIFTSASField(adaptive_scene, raster_granularity=8), 1
    )
    adaptive_pred, _ = render_sas_bins(
        adaptive_field, subset_radii[:2], tx, rx, corners,
        num_rays=16, opacity_scale=0.0, lambertian_ratio=1.0, normal_step=0.01,
    )
    adaptive_loss = adaptive_pred.abs().square().sum()
    adaptive_loss.backward()
    check(torch.isfinite(adaptive_scene.w_re.grad).all() and adaptive_scene.w_re.grad.abs().sum() > 0,
          "adaptive coefficient gradient")
    check(
        adaptive_scene.delta_raw.grad is not None
        and torch.isfinite(adaptive_scene.delta_raw.grad).all()
        and adaptive_scene.delta_raw.grad.norm() > 0,
        "adaptive position gradient is finite and nonzero",
    )

    # A newly activated SH band is invisible to the ordinary order-zero
    # forward path but must receive signal from the zero-forward probe.
    probe_scene = AdaptivePointSHScene.from_regular_grid(
        2, 1.0, device, max_degree=1, init_degree=0, init_scale=0.0,
        capacity=64, compact_sh_eval=True,
    )
    with torch.no_grad():
        probe_scene.w_re[0, 0] = 0.2
        # The next band remains exactly zero, as it must at order zero.  The
        # probe tests whether it nevertheless receives a useful derivative.
        probe_scene.w_re[:, 1:] = 0.0
        probe_scene.w_im[:, 1:] = 0.0
    probe_field = build_adapter(AdaptiveRIFTSASField(probe_scene, raster_granularity=8), 1)
    probe_model_point = torch.tensor([[-0.41, -0.57, -0.46]], dtype=torch.float32)
    probe_points = probe_field.scene_center + probe_field.scene_half_extent * probe_model_point
    probe_directions = normalize(torch.tensor([[0.2, 0.1, 1.0]], dtype=torch.float32))
    ordinary = probe_field.query_sas(probe_points, probe_directions, normal_step=0.01)
    target = torch.tensor([0.35 + 0.2j], dtype=torch.complex64)
    ordinary_loss = (ordinary["scatterer"] - target).abs().square().sum()
    ordinary_grad = torch.autograd.grad(ordinary_loss, probe_scene.w_re, retain_graph=False)[0]
    probe = probe_field.query_sas(
        probe_points, probe_directions, normal_step=0.01, probe_next_band=True
    )
    check(torch.allclose(ordinary["scatterer"], probe["scatterer"], atol=0.0, rtol=0.0),
          "zero-initialized probe leaves forward prediction unchanged")
    probe_loss = (probe["scatterer"] - target).abs().square().sum()
    probe_grad = torch.autograd.grad(probe_loss, probe_scene.w_re, retain_graph=False)[0]
    check(abs(float(ordinary_grad[0, 1])) < 1.0e-10, "next-band ordinary forward is zero")
    check(torch.isfinite(probe_grad).all() and probe_grad[0, 1].abs() > 0,
          "next-band probe gradient is nonzero")

    before_split = adaptive_pred.detach()
    with torch.no_grad():
        adaptive_scene.pos_grad_accum[0] = 1.0
        adaptive_scene.grad_accum_count.fill_(1)
    adaptive_scene.split(
        criterion="position_world", mode="count", count=1, max_level=1,
        in_place_heir=True,
    )
    after_split, _ = render_sas_bins(
        adaptive_field, subset_radii[:2], tx, rx, corners,
        num_rays=16, opacity_scale=0.0, lambertian_ratio=1.0, normal_step=0.01,
    )
    check(torch.allclose(before_split, after_split, atol=1e-6, rtol=0.0),
          "in-place adaptive heir preserves prediction")

    print("All shared SAS gates passed.")


if __name__ == "__main__":
    main()
