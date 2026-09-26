"""Formula-level parity of the tinycudann torch shim with the pinned tiny-cuda-nn
source (ledger gate 1, CPU). Every expected value is re-derived here
independently of ``rift_pvc.tcnn_torch``: grid arithmetic from ``grid.h`` /
``common_device.h``, hashing with Python integers, the SH table's properties,
MLP products from explicit slices."""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rift_pvc import tcnn_torch as tcnn  # noqa: E402
from rift_pvc.tcnn_torch import layout as L  # noqa: E402
from rift_pvc.tcnn_torch.spherical_harmonics import sh_enc  # noqa: E402

PRODUCTION_PLS = float(np.exp2(np.log2(512 * 1 / 16) / (16 - 1)))   # tcnn_utils.get_encoding_config
GRID_CFG = {"otype": "HashGrid", "n_levels": 16, "n_features_per_level": 2, "log2_hashmap_size": 19,
            "base_resolution": 16, "per_level_scale": PRODUCTION_PLS}
SH_CFG = {"otype": "SphericalHarmonics", "degree": 4}
MLP_CFG = {"otype": "FullyFusedMLP", "activation": "ReLU", "output_activation": "None", "n_neurons": 64,
           "n_hidden_layers": 1}
PRIMES = (1, 2654435761, 805459861)


@pytest.fixture(autouse=True)
def fp32_mode(monkeypatch):
    monkeypatch.delenv("RIFT_PVC_TCNN_HALF", raising=False)
    monkeypatch.setenv("RIFT_ACCELERATOR", "cpu")


def reference_levels(n_levels, base, pls, log2_size, dims=3):
    """Independent transcription of the grid.h constructor in numpy float32."""
    scale32 = np.float32(pls)
    log2s = np.log2(scale32)                     # numpy float32 log2f
    out, offset = [], 0
    for level in range(n_levels):
        s = np.exp2(np.float32(level) * log2s) * np.float32(base) - np.float32(1)   # grid_scale
        res = int(np.ceil(s)) + 1                                                    # grid_resolution
        n = res ** dims
        n = ((n + 7) // 8) * 8
        n = min(n, 1 << log2_size)
        out.append((level, float(s), res, n, offset, n < res ** dims))
        offset += n
    return out, offset * 2


def test_grid_layout_matches_the_source_arithmetic_for_the_production_config():
    expected, n_params = reference_levels(16, 16, PRODUCTION_PLS, 19)
    layout = L.grid_layout(3, 16, 2, 19, 16, PRODUCTION_PLS)
    assert layout.n_params == n_params
    for lv, (level, scale, res, n, offset, hashed) in zip(layout.levels, expected):
        assert (lv.level, lv.resolution, lv.params_in_level, lv.offset, lv.hashed) == (level, res, n, offset, hashed)
        assert lv.scale == pytest.approx(scale, rel=1e-6)
    # Facts fixed by the source: 16 vertices at level 0 (4096 entries), hashing
    # only where res**3 exceeds 2**19, tables padded to 8, offsets are prefix sums.
    assert layout.levels[0].resolution == 16 and layout.levels[0].params_in_level == 4096
    assert [lv.hashed for lv in layout.levels] == [lv.resolution ** 3 > 2 ** 19 for lv in layout.levels]
    assert all(lv.params_in_level % 8 == 0 for lv in layout.levels)
    assert all(b.offset == a.offset + a.params_in_level for a, b in zip(layout.levels, layout.levels[1:]))
    assert layout.levels[-1].resolution in (512, 513)   # ceil of 511 +- one float32 ulp, plus one


def test_grid_layout_types_and_limits():
    dense = L.grid_layout(3, 4, 2, 19, 4, 2.0, grid_type="Dense")
    assert [lv.params_in_level for lv in dense.levels] == [64, 512, 4096, 32768] and not any(lv.hashed for lv in dense.levels)
    tiled = L.grid_layout(3, 4, 2, 19, 4, 2.0, grid_type="Tiled")
    assert all(lv.params_in_level == 64 for lv in tiled.levels)
    with pytest.raises(ValueError):
        L.grid_layout(3, 129, 2, 19, 16, 2.0)


def python_grid_index(corner, level):
    """grid_index with uint32 semantics, in Python integers."""
    dims = len(corner)
    if level.hashed:
        index = 0
        for d in range(dims):
            index ^= (corner[d] * PRIMES[d]) & 0xFFFFFFFF
    else:
        index, stride = 0, 1
        for d in range(dims):
            index += corner[d] * stride
            stride *= level.resolution
        index &= 0xFFFFFFFF
    return index % level.params_in_level


def test_hash_and_dense_indices_match_a_direct_evaluation():
    enc = tcnn.Encoding(3, GRID_CFG)
    rng = np.random.default_rng(0)
    for level in (enc.layout.levels[2], enc.layout.levels[7], enc.layout.levels[15]):
        corners = rng.integers(0, level.resolution + 1, size=(64, 3))
        got = enc.level_index(torch.from_numpy(corners), level).tolist()
        assert got == [python_grid_index(tuple(int(v) for v in c), level) for c in corners]
    # the edge corner (resolution) wraps exactly as uint32 arithmetic does
    level = enc.layout.levels[0]
    assert enc.level_index(torch.tensor([[16, 15, 15]]), level).item() == python_grid_index((16, 15, 15), level)


def python_encode(enc, x):
    """kernel_grid transcribed with Python floats: fmaf(scale, x, 0.5), floor, trilinear."""
    table = enc.params.detach().view(-1, enc.layout.n_features_per_level).numpy()
    outputs = []
    for level in enc.layout.levels:
        pos = [np.float32(np.float64(np.float32(v)) * level.scale + 0.5) for v in x]
        grid = [int(math.floor(p)) for p in pos]
        frac = [float(p - np.float32(g)) for p, g in zip(pos, grid)]
        acc = np.zeros(enc.layout.n_features_per_level, dtype=np.float64)
        for idx in range(8):
            weight, corner = 1.0, []
            for dim in range(3):
                if idx & (1 << dim):
                    weight *= frac[dim]; corner.append(grid[dim] + 1)
                else:
                    weight *= 1 - frac[dim]; corner.append(grid[dim])
            acc += weight * table[level.offset + python_grid_index(corner, level)]
        outputs.append(acc)
    return np.concatenate(outputs)


def test_trilinear_forward_matches_a_direct_evaluation():
    enc = tcnn.Encoding(3, {"otype": "HashGrid", "n_levels": 4, "n_features_per_level": 2, "log2_hashmap_size": 10,
                            "base_resolution": 4, "per_level_scale": 2.0})
    assert [lv.hashed for lv in enc.layout.levels] == [False, False, True, True]
    x = torch.rand(32, 3)
    x[0] = torch.tensor([0.0, 0.0, 0.0]); x[1] = torch.tensor([1.0, 1.0, 1.0])   # edges included
    out = enc(x).detach().numpy()
    for i in range(32):
        np.testing.assert_allclose(out[i], python_encode(enc, x[i].tolist()), rtol=1e-5, atol=1e-9)


def test_grid_gradients_match_finite_differences_and_flow_to_params():
    enc = tcnn.Encoding(3, {"otype": "HashGrid", "n_levels": 3, "n_features_per_level": 2, "log2_hashmap_size": 12,
                            "base_resolution": 8, "per_level_scale": 1.5})
    eps = 1e-4
    generator = torch.Generator().manual_seed(4)
    # d(out)/dx is piecewise constant (scale * feature differences): keep every probe at least 4*eps away
    # from a cell boundary at every level, so the central differences stay inside one cell.
    points = []
    while len(points) < 6:
        candidate = torch.rand(3, generator=generator)
        fractions = torch.stack([(candidate * lv.scale + 0.5) % 1.0 for lv in enc.layout.levels])
        margin = max(4 * eps * lv.scale for lv in enc.layout.levels)
        if ((fractions > margin) & (fractions < 1 - margin)).all():
            points.append(candidate)
    x = torch.stack(points).requires_grad_(True)
    weights = torch.randn(6, enc.n_output_dims, generator=generator)
    (enc(x) * weights).sum().backward()
    for dim in range(3):
        shifted = x.detach().clone(); shifted[:, dim] += eps
        base = x.detach().clone(); base[:, dim] -= eps
        fd = ((enc(shifted) * weights).sum(1) - (enc(base) * weights).sum(1)) / (2 * eps)
        np.testing.assert_allclose(x.grad[:, dim].numpy(), fd.detach().numpy(), rtol=2e-2, atol=2e-4)
    assert enc.params.grad is not None and (enc.params.grad != 0).sum() > 0
    assert enc.params.grad.shape == enc.params.shape


def test_initialisation_is_uniform_1e_4_and_seeded():
    a, b = tcnn.Encoding(3, GRID_CFG, seed=1337), tcnn.Encoding(3, GRID_CFG, seed=1337)
    c = tcnn.Encoding(3, GRID_CFG, seed=7)
    assert torch.equal(a.params, b.params) and not torch.equal(a.params, c.params)
    assert a.params.abs().max() <= 1e-4 and a.params.abs().max() > 0.9e-4
    assert a.params.dtype == torch.float32 and a.params.std().item() == pytest.approx(2e-4 / math.sqrt(12), rel=2e-2)


def test_spherical_harmonics_table_mapping_and_orthonormality():
    sh = tcnn.Encoding(3, SH_CFG)
    assert sh.n_output_dims == 16 and sh.params.numel() == 0 and isinstance(sh.params, torch.nn.Parameter)
    d = torch.nn.functional.normalize(torch.randn(2048, 3), dim=-1)
    # kernel_sh maps the input as 2x-1: feeding (d+1)/2 evaluates the table at d
    x = torch.rand(2048, 3)
    torch.testing.assert_close(sh(x), sh_enc(4, x * 2.0 - 1.0), rtol=0, atol=0)
    out = sh((d + 1) / 2)   # (d+1)/2 rounds in float32, hence a tolerance
    torch.testing.assert_close(out, sh_enc(4, d), rtol=1e-3, atol=1e-6)
    x, y, z = d.unbind(-1)
    expected_l1 = torch.stack([torch.full_like(x, 0.28209479177387814), -0.48860251190291987 * y,
                               0.48860251190291987 * z, -0.48860251190291987 * x], -1)
    torch.testing.assert_close(out[:, :4], expected_l1, rtol=1e-6, atol=1e-7)
    # Real SH of degree < 4 are orthonormal on the sphere: Monte Carlo Gram matrix ~ identity.
    big = torch.nn.functional.normalize(torch.randn(400000, 3, dtype=torch.float64), dim=-1).float()
    values = sh_enc(4, big).double()
    gram = values.t() @ values / len(big) * (4 * math.pi)
    assert torch.allclose(gram, torch.eye(16, dtype=torch.float64), atol=3e-2)
    # a random direction: coefficients are the known l=2 forms
    torch.testing.assert_close(out[:, 4], 1.0925484305920792 * x * y, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(out[:, 6], 0.94617469575755997 * z * z - 0.31539156525251999, rtol=1e-6, atol=1e-7)


def test_spherical_harmonics_gradient_matches_finite_differences():
    sh = tcnn.Encoding(3, SH_CFG)
    x = torch.rand(5, 3, dtype=torch.float64).float().requires_grad_(True)
    w = torch.randn(5, 16)
    (sh(x) * w).sum().backward()
    eps = 1e-3
    for dim in range(3):
        plus, minus = x.detach().clone(), x.detach().clone()
        plus[:, dim] += eps; minus[:, dim] -= eps
        fd = ((sh(plus) * w).sum(1) - (sh(minus) * w).sum(1)) / (2 * eps)
        np.testing.assert_allclose(x.grad[:, dim].numpy(), fd.detach().numpy(), rtol=1e-3, atol=1e-3)


def test_mlp_layout_products_padding_and_truncation():
    cfg = dict(MLP_CFG, n_hidden_layers=2)
    net = tcnn.Network(20, 3, cfg)
    lay = net.layout
    assert (lay.padded_input_width, lay.padded_output_width) == (32, 16)
    assert [(m.name, m.rows, m.cols) for m in lay.matrices] == [("input", 64, 32), ("hidden0", 64, 64), ("output", 16, 64)]
    assert lay.n_params == 64 * 32 + 64 * 64 + 16 * 64 == net.params.numel()
    p = net.params.detach()
    w_in = p[:64 * 32].view(64, 32); w_h = p[64 * 32:64 * 32 + 64 * 64].view(64, 64); w_out = p[64 * 32 + 64 * 64:].view(16, 64)
    x = torch.randn(9, 20)
    padded = torch.cat([x, torch.ones(9, 12)], 1)                      # identity encoding pads with ones
    h = torch.relu(padded @ w_in.t()); h = torch.relu(h @ w_h.t()); y = (h @ w_out.t())[:, :3]
    torch.testing.assert_close(net(x), y, rtol=1e-6, atol=1e-6)
    assert net(x).shape == (9, 3) and net.n_output_dims == 3


def test_mlp_xavier_initialisation_per_matrix_in_order():
    net = tcnn.Network(48, 1, MLP_CFG)
    assert [(m.rows, m.cols) for m in net.layout.matrices] == [(64, 48), (16, 64)]
    for matrix, weight in zip(net.layout.matrices, net.weight_matrices()):
        bound = math.sqrt(6.0 / (matrix.rows + matrix.cols))
        assert matrix.xavier_bound == pytest.approx(bound, rel=1e-6)
        assert weight.abs().max() <= bound and weight.abs().max() > 0.95 * bound
        assert weight.std().item() == pytest.approx(bound / math.sqrt(3), rel=0.1)
    same = tcnn.Network(48, 1, MLP_CFG)
    assert torch.equal(net.params, same.params)
    # xyz_net and alpha_net of the release share shapes and the default seed: identical initial weights, as with TCNN
    assert torch.equal(tcnn.Network(32, 32, MLP_CFG).weight_matrices()[0], tcnn.Network(32, 1, MLP_CFG).weight_matrices()[0])


def test_activations_follow_the_source_formulas():
    from rift_pvc.tcnn_torch.fully_fused_mlp import apply_activation
    x = torch.linspace(-3, 3, 13)
    torch.testing.assert_close(apply_activation("Softplus", x), torch.log(torch.exp(10 * x) + 1) / 10)
    torch.testing.assert_close(apply_activation("Squareplus", x), 0.5 * (10 * x + torch.sqrt((10 * x) ** 2 + 4)) / 10)
    torch.testing.assert_close(apply_activation("LeakyReLU", x), torch.where(x > 0, x, 0.01 * x))
    assert torch.equal(apply_activation("None", x), x)
    net = tcnn.Network(16, 1, dict(MLP_CFG, activation="relu", output_activation="sigmoid"))
    assert (net.activation, net.output_activation) == ("ReLU", "Sigmoid")
    assert ((net(torch.randn(4, 16)) > 0) & (net(torch.randn(4, 16)) < 1)).all()


def test_production_module_shapes_and_param_counts():
    enc, sh = tcnn.Encoding(3, GRID_CFG), tcnn.Encoding(3, SH_CFG)
    xyz_net = tcnn.Network(enc.n_output_dims, 32, MLP_CFG)
    alpha_net = tcnn.Network(32, 1, MLP_CFG)
    rd_net = tcnn.Network(sh.n_output_dims + 32, 1, MLP_CFG)
    assert (enc.n_output_dims, sh.n_output_dims) == (32, 16)
    assert [m.params.numel() for m in (enc, sh, xyz_net, alpha_net, rd_net)] == [10523376, 0, 4096, 3072, 4096]
    assert (xyz_net.padded_output_width, alpha_net.padded_output_width) == (32, 16)
    assert list(enc.layout.slices())[:2] == ["level0", "level1"] and enc.layout.slices()["level1"] == (8192, (9264, 2))
    assert xyz_net.layout.slices() == {"input": (0, (64, 32)), "output": (2048, (32, 64))}


def test_module_api_surface_and_state_dict():
    net = tcnn.Network(32, 1, MLP_CFG)
    assert list(net.state_dict()) == ["params"] and list(dict(net.named_parameters())) == ["params"]
    assert net.native_tcnn_module is None and net.seed == 1337 and net.dtype == torch.float32 and net.loss_scale == 1.0
    assert "n_input_dims=32" in repr(net) and tcnn.free_temporary_memory() is None and tcnn.supports_jit_fusion() is False
    enc16 = tcnn.Encoding(3, GRID_CFG, dtype=torch.float16)
    assert enc16(torch.rand(4, 3)).dtype == torch.float16 and enc16.params.dtype == torch.float32
    assert tcnn.Encoding(3, SH_CFG, dtype=torch.float32)(torch.rand(2, 3)).dtype == torch.float32
    with pytest.raises(ValueError):
        tcnn.Encoding(3, SH_CFG, dtype=torch.float64)
    ident = tcnn.identity()
    assert ident["model_backend"] == "upstream-tcnn-torchshim" and ident["version"] == tcnn.SHIM_VERSION == "torchshim-2" and ident["precision"] == "fp32"


def test_half_parity_mode(monkeypatch):
    monkeypatch.setenv("RIFT_PVC_TCNN_HALF", "1")
    assert tcnn.precision_mode() == "fp16"
    enc, net, sh = tcnn.Encoding(3, GRID_CFG), tcnn.Network(32, 1, MLP_CFG), tcnn.Encoding(3, SH_CFG)
    assert (enc.dtype, net.dtype, sh.dtype) == (torch.float16,) * 3 and net.loss_scale == 128.0
    x = torch.rand(64, 3)
    out16 = enc(x)
    assert out16.dtype == torch.float16
    monkeypatch.delenv("RIFT_PVC_TCNN_HALF")
    enc32 = tcnn.Encoding(3, GRID_CFG)
    torch.testing.assert_close(out16.float(), enc32(x), rtol=2e-3, atol=2e-7)
    y16 = net(torch.randn(8, 32))
    assert y16.dtype == torch.float16 and y16.shape == (8, 1)


def test_unsupported_configurations_raise():
    with pytest.raises(ValueError):
        tcnn.Encoding(3, {"otype": "Frequency", "degree": 4})
    with pytest.raises(NotImplementedError):
        tcnn.Encoding(3, dict(GRID_CFG, interpolation="Smoothstep"))
    with pytest.raises(NotImplementedError):
        tcnn.Encoding(3, {"otype": "SphericalHarmonics", "degree": 5})
    with pytest.raises(RuntimeError):
        tcnn.Encoding(2, SH_CFG)
    with pytest.raises(ValueError):
        tcnn.Network(3, 1, dict(MLP_CFG, n_neurons=48))
    with pytest.raises(ValueError):
        tcnn.Network(3, 1, dict(MLP_CFG, otype="CutlassMLP"))
    with pytest.raises(ValueError):
        tcnn.Network(3, 1, dict(MLP_CFG, n_hidden_layers=0))
    with pytest.raises(ValueError):
        tcnn.Encoding(3, dict(GRID_CFG, hash="Prime"))


def test_network_with_input_encoding_layout_and_padding():
    nwie = tcnn.NetworkWithInputEncoding(3, 1, {"otype": "SphericalHarmonics", "degree": 3}, MLP_CFG)
    assert nwie.params.numel() == 64 * 16 + 16 * 64 and list(nwie.state_dict()) == ["params"]
    d = torch.rand(5, 3)
    encoded = sh_enc(3, d * 2 - 1)
    padded = torch.cat([torch.ones(5, 7), encoded], 1)                # kernel_sh: ones first
    w_in = nwie.params[:64 * 16].view(64, 16); w_out = nwie.params[64 * 16:].view(16, 64)
    expected = (torch.relu(padded @ w_in.t()) @ w_out.t())[:, :1]
    torch.testing.assert_close(nwie(d), expected, rtol=1e-6, atol=1e-6)
    nwie(d).sum().backward()
    assert nwie.params.grad is not None and nwie.params.grad.abs().sum() > 0
    grid = tcnn.NetworkWithInputEncoding(3, 2, {"otype": "HashGrid", "n_levels": 3, "n_features_per_level": 2,
                                                "log2_hashmap_size": 8, "base_resolution": 4}, MLP_CFG)
    assert grid.n_pad == 10 and grid(d).shape == (5, 2)


def test_weight_gradient_strategies_agree_with_autograd_and_each_other(monkeypatch):
    from rift_pvc.tcnn_torch.linear import linear, weight_gradient, strategy
    assert strategy() == "bmm:256"
    x = torch.randn(1000, 48, dtype=torch.float32)        # not a multiple of the block: exercises the zero padding
    w = torch.randn(64, 48, dtype=torch.float32)
    g = torch.randn(1000, 64)
    reference = (g.double().t() @ x.double())
    for how in ("gemm", "fp64", "bmm:256", "bmm:64", "outer:128"):
        got = weight_gradient(g, x, how)
        assert got.dtype == torch.float32 and got.shape == (64, 48)
        assert float((got.double() - reference).abs().max() / reference.abs().max()) < 1e-6, how
        xa, wa = x.clone().requires_grad_(True), w.clone().requires_grad_(True)
        (linear(xa, wa, how) * g).sum().backward()
        assert float((wa.grad.double() - reference).abs().max() / reference.abs().max()) < 1e-6, how
        torch.testing.assert_close(xa.grad, g @ w, rtol=1e-5, atol=1e-5)
    monkeypatch.setenv("RIFT_PVC_TCNN_WEIGHT_GRAD", "gemm")
    assert strategy() == "gemm" and tcnn.identity()["weight_grad"] == "gemm"
    with pytest.raises(ValueError):
        weight_gradient(g, x, "nonsense")
    net = tcnn.Network(48, 1, MLP_CFG)
    assert net.weight_grad == "gemm"
    # the release's MLPs train identically under every strategy up to float32 rounding
    monkeypatch.delenv("RIFT_PVC_TCNN_WEIGHT_GRAD")
    a, b = tcnn.Network(48, 1, MLP_CFG), tcnn.Network(48, 1, MLP_CFG)
    b.weight_grad = "gemm"
    inp = torch.randn(3000, 48)
    a(inp).sum().backward(); b(inp).sum().backward()
    torch.testing.assert_close(a.params.grad, b.params.grad, rtol=1e-5, atol=1e-5)   # float32 summation order


def test_half_mode_loss_scaling_keeps_small_gradients_from_underflowing(monkeypatch):
    """TCNN scales dL/dy by 128 before its fp16 backward; the shim's fp16 mode does the same."""
    monkeypatch.delenv("RIFT_PVC_TCNN_HALF", raising=False)
    full = {"grid": tcnn.Encoding(3, GRID_CFG), "sh": tcnn.Encoding(3, SH_CFG), "mlp": tcnn.Network(32, 1, MLP_CFG)}
    monkeypatch.setenv("RIFT_PVC_TCNN_HALF", "1")
    half = {"grid": tcnn.Encoding(3, GRID_CFG), "sh": tcnn.Encoding(3, SH_CFG), "mlp": tcnn.Network(32, 1, MLP_CFG)}
    assert tcnn.identity()["fp16_loss_scale"] == 128.0 and half["grid"].loss_scale == 128.0
    with torch.no_grad():
        full["grid"].params.mul_(64.0)                  # features in fp16's normal range, as in the @amp parity records
        for name in full:
            half[name].params.copy_(full[name].params)
    inputs = {"grid": torch.rand(1024, 3), "sh": torch.rand(1024, 3), "mlp": torch.rand(1024, 32) * 1e-2}
    for name in full:
        grads = {}
        for mode, module in (("fp32", full[name]), ("fp16", half[name])):
            x = inputs[name].clone().requires_grad_(True)
            module.zero_grad(set_to_none=True)
            module(x).float().mean().backward()   # per-element upstream gradient ~3e-5: fp16-subnormal without the x128 scaling
            grads[mode] = (x.grad.clone(), module.params.grad.clone() if module.params.numel() else None)
        for kind, (a, b) in enumerate(zip(grads["fp32"], grads["fp16"])):
            if a is None:
                continue
            assert b.abs().sum() > 0, f"{name}: fp16-mode gradient underflowed to zero"
            err = float((a - b).norm() / a.norm().clamp_min(1e-30))
            assert err < 5e-2, f"{name}: {'input' if kind == 0 else 'param'} gradient fp16 vs fp32 rel {err:.3e}"
