#!/usr/bin/env python
"""CPU validation gates for the independent SH-SAS implementation."""

from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rift.sh_sas import (  # noqa: E402
    PAPER_HASH_BASE_RESOLUTION,
    PAPER_HASH_FINAL_RESOLUTION,
    PAPER_HASH_LEVELS,
    PAPER_MLP_HIDDEN_LAYERS,
    PAPER_MLP_WIDTH,
    PAPER_SH_DEGREE,
    SHSASField,
    Y00,
    lambertian_cosine,
    real_sh_basis_for_directions,
)
from rift.config import cc  # noqa: E402
from rift.range_operator import range_forward_operator  # noqa: E402
from train_sh_sas import combine_metrics, metric_record, parse_args  # noqa: E402


PASSED = 0


def gate(name, condition):
    global PASSED
    if not bool(condition):
        raise AssertionError(name)
    PASSED += 1
    print(f"PASS {PASSED:02d}: {name}")


def tiny_model() -> SHSASField:
    torch.manual_seed(7)
    return SHSASField(
        extent=0.5,
        granularity=4,
        sh_degree=3,
        hidden_dim=8,
        hash_levels=3,
        hash_features=2,
        hash_base_resolution=2,
        hash_final_resolution=8,
        hash_log2_size=6,
        device=torch.device("cpu"),
    )


def main():
    gate("paper fixes degree three", PAPER_SH_DEGREE == 3)
    gate(
        "paper network constants are 16 levels, base 16, final 4096, two-by-32 MLP",
        (
            PAPER_HASH_LEVELS,
            PAPER_HASH_BASE_RESOLUTION,
            PAPER_HASH_FINAL_RESOLUTION,
            PAPER_MLP_HIDDEN_LAYERS,
            PAPER_MLP_WIDTH,
        )
        == (16, 16, 4096, 2, 32),
    )

    defaults = parse_args(["--npz-path", "dummy.npz"])
    gate(
        "training CLI defaults select the paper architecture and DC opacity",
        defaults.sh_degree == 3
        and defaults.hash_levels == 16
        and defaults.hash_final_resolution == 4096
        and defaults.hidden_dim == 32
        and defaults.opacity_key == "dc",
    )
    gate(
        "simulated-data defaults disable every paper prior",
        defaults.sparse_weight == defaults.density_tv_weight
        == defaults.scatter_tv_weight
        == defaults.phase_tv_weight
        == 0.0,
    )

    model = tiny_model()
    coeff = model.dense_coefficients(chunk_size=17)
    gate("hash field emits a dense 4^3 by 16 complex coefficient lattice", coeff.shape == (4, 4, 4, 16) and coeff.is_complex())
    gate("hash field is nonzero under random neural initialization", float(coeff.abs().max()) > 0.0)

    loss = coeff.abs().square().mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    gate("coefficient query backpropagates into hash and MLP parameters", any(g is not None and torch.isfinite(g).all() and float(g.abs().sum()) > 0 for g in grads))
    model.zero_grad(set_to_none=True)

    synthetic = torch.zeros(4, 4, 4, 16, dtype=torch.complex64)
    synthetic[..., 0] = 2.0 + 3.0j
    expected = math.sqrt(13.0) / math.sqrt(4.0 * math.pi)
    gate("DC density is exactly abs(c00)/sqrt(4pi)", torch.allclose(model.dc_amplitude(synthetic), torch.full((4, 4, 4), expected), atol=1e-6))

    x = torch.arange(4, dtype=torch.float32).view(4, 1, 1).expand(4, 4, 4)
    gradient_coeff = torch.zeros_like(synthetic)
    gradient_coeff[..., 0] = torch.complex(x + 1.0, torch.zeros_like(x))
    normals = model.normals_from_dc(gradient_coeff)
    gate("negative DC gradient produces outward -x normals", torch.allclose(normals[1:3, :, :, 0], -torch.ones_like(normals[1:3, :, :, 0]), atol=1e-5))

    directions = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    basis0 = real_sh_basis_for_directions(directions, 0)
    gate("degree-zero SH basis is direction-independent Y00", torch.allclose(basis0[:, 0], torch.full((2,), Y00), atol=1e-7))

    points = torch.tensor([[0.0, 0.0, 0.0]])
    normal = torch.tensor([[0.0, 0.0, 1.0]])
    gate("Lambertian term accepts transmitter-facing normals", torch.allclose(lambertian_cosine(points, normal, torch.tensor([0.0, 0.0, 2.0])), torch.ones(1)))
    gate("Lambertian term rejects back-facing normals", torch.allclose(lambertian_cosine(points, -normal, torch.tensor([0.0, 0.0, 2.0])), torch.zeros(1)))

    tx = torch.tensor([[-0.1, 0.0, 2.0], [0.1, 0.0, 2.0]])
    rx = torch.tensor([[-0.1, 0.0, 2.0], [0.1, 0.0, 2.0]])
    view_clear = model.view_field(
        tx, rx, opacity_scale=0.0, use_lambertian=False, use_occlusion=True,
        query_chunk=19, occlusion_point_chunk=23,
    )
    gate("zeta zero recovers unit transmittance exactly", torch.equal(view_clear["transmittance"], torch.ones_like(view_clear["transmittance"])))
    gate("without Lambertian/occlusion, rendered weights equal the SH field", torch.allclose(view_clear["weights"].reshape(4, 4, 4), view_clear["scattering"], atol=1e-7))

    view_occ = model.view_field(
        tx, rx, opacity_scale=0.1, opacity_key="dc", opacity_normalize=True,
        use_lambertian=True, use_occlusion=True, query_chunk=19,
        occlusion_point_chunk=23,
    )
    gate("separate Tx/Rx transmittance is finite and lies in [0,1]", torch.isfinite(view_occ["transmittance"]).all() and float(view_occ["transmittance"].min()) >= 0.0 and float(view_occ["transmittance"].max()) <= 1.0)
    gate("Lambertian factor is finite and lies in [0,1]", torch.isfinite(view_occ["lambertian"]).all() and float(view_occ["lambertian"].min()) >= 0.0 and float(view_occ["lambertian"].max()) <= 1.0 + 1e-6)

    view_literal = model.view_field(
        tx, rx, opacity_scale=0.25, opacity_key="dc", opacity_normalize=False,
        use_lambertian=False, use_occlusion=True, query_chunk=19,
        occlusion_point_chunk=23,
    )
    gate("--no-opacity-normalize recovers literal Eq. 4 rho=zeta*abs(c00)/sqrt(4pi)", torch.allclose(view_literal["opacity"], 0.25 * view_literal["density"], rtol=1e-6, atol=1e-9))

    regs = model.regularizers(view_occ)
    gate("all Eq. 8 regularizers are finite and non-negative", all(torch.isfinite(value) and float(value) >= 0.0 for value in regs.values()))
    objective = view_occ["weights"].abs().square().mean() + sum(regs.values())
    objective.backward()
    gate("full SH/Lambertian/transmittance path is differentiable", any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    model.zero_grad(set_to_none=True)
    operator_view = model.view_field(
        tx[:1], rx[:1], opacity_scale=0.0, use_lambertian=False,
        use_occlusion=False, query_chunk=19,
    )
    frequencies = torch.linspace(9.0e9, 9.3e9, 6, dtype=torch.float32)
    kvector = 2.0 * torch.pi * frequencies / cc
    rendered = range_forward_operator(
        frequencies, kvector, rx[:1], tx[:1], operator_view["points"],
        operator_view["weights"], freq_indices=torch.tensor([0, 2, 5]),
        oversample=2, kernel_width=4, pair_chunk=1, point_chunk=17,
        compute_dtype=torch.float64, range_model="sum2",
    )
    gate("exact bistatic radar adapter returns the requested [freq,rx,tx] cube", rendered.shape == (3, 1, 1) and torch.isfinite(rendered).all())
    rendered.abs().square().mean().backward()
    gate("exact radar operator backpropagates into the hash-SH field", any(p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.abs().sum()) > 0 for p in model.parameters()))

    compatibility = model.geometry_compatibility_state(chunk_size=13)
    gate("checkpoint compatibility exports dense real/imag SH tensors", compatibility["w_re"].shape == (4, 4, 4, 16) and compatibility["w_im"].shape == (4, 4, 4, 16))

    target = torch.randn(3, 2, 2, dtype=torch.complex64)
    metrics = combine_metrics([metric_record(target, target)])
    gate("coherent metric plumbing gives zero error for a perfect prediction", metrics["rel_mse"] == 0.0 and metrics["l1_real"] == 0.0 and metrics["mse_abs"] == 0.0)

    gate("paper contract never accepts geometry supervision", not hasattr(defaults, "stl") and not hasattr(defaults, "point_cloud"))
    print(f"SH-SAS validation passed: {PASSED}/{PASSED} gates")


if __name__ == "__main__":
    main()
