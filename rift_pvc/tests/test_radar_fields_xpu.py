"""Radar Fields PVC device checks (XPU; CUDA when RIFT_PVC_ALLOW_BACKEND=cuda).

Run with ``sbatch --export=ALL,PYTEST_TARGET=rift_pvc/tests/test_radar_fields_xpu.py
scripts_pvc/run_tests_pvc.sbatch``. Any ``Aten Op fallback from XPU to CPU``
line in the job log is a defect (checked by the launcher, not here)."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rift_pvc import accelerator  # noqa: E402
from rift_pvc import tcnn_torch as tcnn  # noqa: E402
from rift_pvc import radar_fields_training as twins  # noqa: E402

pytestmark = pytest.mark.skipif(not accelerator.is_available(), reason="no XPU/CUDA device in this process")
REFERENCE = ROOT / "external" / "RadarFields_reference"
needs_reference = pytest.mark.skipif(not REFERENCE.is_dir(), reason="pinned Radar Fields checkout not present")

PLS = float(np.exp2(np.log2(512 / 16) / 15))
GRID_CFG = {"otype": "HashGrid", "n_levels": 16, "n_features_per_level": 2, "log2_hashmap_size": 19,
            "base_resolution": 16, "per_level_scale": PLS}
SH_CFG = {"otype": "SphericalHarmonics", "degree": 4}
MLP_CFG = {"otype": "FullyFusedMLP", "activation": "ReLU", "output_activation": "None", "n_neurons": 64, "n_hidden_layers": 1}


@pytest.fixture
def device():
    return accelerator.device(0)


def _modules():
    enc, sh = tcnn.Encoding(3, GRID_CFG), tcnn.Encoding(3, SH_CFG)
    return {"encode_xyz": enc, "encode_angle": sh, "xyz_net": tcnn.Network(32, 32, MLP_CFG),
            "alpha_net": tcnn.Network(32, 1, MLP_CFG), "rd_net": tcnn.Network(48, 1, MLP_CFG)}


def _inputs(name, n=4096):
    g = torch.Generator().manual_seed(3)
    if name == "encode_xyz":
        return torch.rand(n, 3, generator=g)
    if name == "encode_angle":
        return torch.nn.functional.normalize(torch.randn(n, 3, generator=g), dim=-1)
    width = {"xyz_net": 32, "alpha_net": 32, "rd_net": 48}[name]
    return torch.randn(n, width, generator=g) * 1e-2


@pytest.mark.parametrize("name", ("encode_xyz", "encode_angle", "xyz_net", "alpha_net", "rd_net"))
def test_shim_modules_agree_with_cpu_on_the_device(device, name, monkeypatch):
    monkeypatch.delenv("RIFT_PVC_TCNN_HALF", raising=False)
    cpu = _modules()[name]
    dev = _modules()[name].to(device)
    x = _inputs(name).requires_grad_(True)
    xd = x.detach().to(device).requires_grad_(True)
    out_cpu, out_dev = cpu(x), dev(xd)
    torch.testing.assert_close(out_dev.cpu(), out_cpu, rtol=1e-5, atol=1e-7)
    out_cpu.sum().backward(); out_dev.sum().backward()
    if cpu.params.numel():
        # float32 reduction order differs across devices: near-zero gradient entries need an absolute floor
        torch.testing.assert_close(dev.params.grad.cpu(), cpu.params.grad, rtol=1e-4, atol=1e-7)
    torch.testing.assert_close(xd.grad.cpu(), x.grad, rtol=1e-4, atol=1e-6)
    assert out_dev.device.type == device.type


def test_half_mode_runs_on_the_device_and_tracks_fp32(device, monkeypatch):
    monkeypatch.setenv("RIFT_PVC_TCNN_HALF", "1")
    half = {k: m.to(device) for k, m in _modules().items()}
    monkeypatch.delenv("RIFT_PVC_TCNN_HALF")
    full = {k: m.to(device) for k, m in _modules().items()}
    for name in half:
        x = _inputs(name, 1024).to(device)
        a, b = half[name](x), full[name](x)
        assert a.dtype == torch.float16 and b.dtype == torch.float32
        err = float((a.float() - b).norm() / b.norm().clamp_min(1e-30))
        print(f"{name}: fp16 vs fp32 shim rel {err:.3e}")
        assert err < 1e-2


def test_device_rng_fork_and_checkpoint_payload_round_trip(device):
    kind = device.type
    module = torch.xpu if kind == "xpu" else torch.cuda
    torch.manual_seed(11)
    before = torch.rand(4, device=device)
    with torch.random.fork_rng(devices=list(range(module.device_count())), device_type=kind):
        torch.manual_seed(0)
        torch.rand(4, device=device)
    after = torch.rand(4, device=device)
    torch.manual_seed(11); torch.rand(4, device=device)
    torch.testing.assert_close(torch.rand(4, device=device), after)   # the fork did not perturb the stream
    assert not torch.equal(before, after)
    state = module.get_rng_state_all()
    normalized = twins.normalize_device_rng_state(state, expected_device_count=module.device_count(),
                                                  require_present=True, backend=kind)
    draw = torch.rand(3, device=device)
    module.set_rng_state_all(normalized)
    torch.testing.assert_close(torch.rand(3, device=device), draw)
    with pytest.raises(ValueError):
        twins.normalize_device_rng_state(None, require_present=True, backend=kind)
    with pytest.raises(ValueError):
        twins.normalize_device_rng_state(state + state, expected_device_count=module.device_count(), backend=kind)


@needs_reference
def test_original_radarfield_runs_on_the_device_through_the_shim(device, monkeypatch):
    monkeypatch.delenv("RIFT_PVC_TCNN_HALF", raising=False)
    monkeypatch.setenv("RIFT_PVC_TCNN_SHIM", "1")
    monkeypatch.setenv("RIFT_PVC_ALLOW_BACKEND", "cpu,cuda,xpu")
    import train_radar_fields_pvc as pvc
    pvc.install()
    args = pvc.parse_args(["--recipe", "source-adapted-v3", "--npz-path", "unused.npz", "--device", str(device)])
    assert args.model_backend == "upstream-tcnn-torchshim"
    model = twins.build_model(args, device)
    assert next(model.parameters()).device.type == device.type
    model.train()
    xyz = (torch.rand(32768, 3, device=device) - .5) * args.extent
    direction = torch.nn.functional.normalize(torch.randn_like(xyz), dim=-1)
    accelerator.synchronize()
    started = time.time()
    out = model(xyz, direction, mask_progress=.8)
    out["rcs"].square().mean().backward()
    accelerator.synchronize()
    elapsed = time.time() - started
    assert all(torch.isfinite(out[k]).all() for k in out)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads) and any(g.abs().max() > 0 for g in grads)
    print(f"RadarField fwd+bwd on {device} for 32768 queries: {elapsed * 1e3:.1f} ms (first call includes JIT)")
    accelerator.synchronize(); started = time.time()
    out = model(xyz, direction, mask_progress=.8); out["rcs"].square().mean().backward()
    accelerator.synchronize()
    print(f"RadarField fwd+bwd on {device} for 32768 queries, warm: {(time.time() - started) * 1e3:.1f} ms")
    model.eval()
    with torch.no_grad():
        chunked = model.query_chunked(xyz, direction, mask_progress=.8, chunk_size=4099)
        whole = model(xyz, direction, mask_progress=.8)
    for key in whole:
        torch.testing.assert_close(chunked[key], whole[key], rtol=1e-5, atol=1e-6)


def test_grid_backward_determinism_report(device, monkeypatch):
    """Two identical backward passes: is the hash-grid parameter gradient bit-identical on the device?

    Informational for the resume record: the grid gradient is accumulated by
    index_add_ (atomic on XPU, as tiny-cuda-nn's atomicAdd is on CUDA), so a
    resumed trajectory can differ in the last bits even with complete state."""
    monkeypatch.delenv("RIFT_PVC_TCNN_HALF", raising=False)
    enc = tcnn.Encoding(3, GRID_CFG).to(device)
    x = _inputs("encode_xyz", 65536).to(device)
    w = torch.randn(65536, 32, device=device)
    grads, outs = [], []
    for _ in range(3):
        enc.zero_grad(set_to_none=True)
        out = enc(x)
        (out * w).sum().backward()
        outs.append(out.detach().clone()); grads.append(enc.params.grad.detach().clone())
    assert all(torch.equal(outs[0], o) for o in outs[1:])   # forward is deterministic
    diff = max(float((grads[0] - g).abs().max()) for g in grads[1:])
    scale = float(grads[0].abs().max())
    print(f"hash-grid backward on {device}: max |grad diff| over repeats {diff:.3e} (max |grad| {scale:.3e}); "
          f"{'bit-identical' if diff == 0 else 'order-nondeterministic accumulation'}")


def test_accumulation_operator_determinism_report(device):
    """Which XPU accumulation kernels are order-deterministic? (informational for the resume record)."""
    g = torch.Generator().manual_seed(5)
    n, m = 2_000_000, 4096
    values = torch.randn(n, generator=g).to(device)
    index = torch.randint(0, m, (n,), generator=g).to(device)
    def repeat(fn):
        first = fn()
        return max(float((first - fn()).abs().max()) for _ in range(3)), float(first.abs().max())
    reports = {
        "index_add_ (float32)": lambda: torch.zeros(m, device=device).index_add_(0, index, values),
        "index_add_ (float64)": lambda: torch.zeros(m, device=device, dtype=torch.float64).index_add_(0, index, values.double()),
        "scatter_add_ (float32)": lambda: torch.zeros(m, device=device).scatter_add_(0, index, values),
        "index_put_ accumulate (float32)": lambda: torch.zeros(m, device=device).index_put_((index,), values, accumulate=True),
        "bincount weights (float32)": lambda: torch.bincount(index, weights=values, minlength=m),
        "sum over gathered rows (float32)": lambda: values.view(-1, 500).sum(1),
    }
    for name, fn in reports.items():
        diff, scale = repeat(fn)
        print(f"{name} on {device}: max |diff| over repeats {diff:.3e} (max |value| {scale:.3e}) -> "
              f"{'deterministic' if diff == 0 else 'ORDER-NONDETERMINISTIC'}")


def test_reduction_and_gemm_determinism_report(device):
    """Large-K GEMMs (weight gradients), BatchNorm statistics and global reductions on the device."""
    g = torch.Generator().manual_seed(9)
    n = 1_000_000
    h = torch.randn(n, 64, generator=g).to(device)
    grad = torch.randn(n, 32, generator=g).to(device)
    v = torch.randn(n, generator=g).to(device)
    w = torch.randn(64, 32, generator=g).to(device).requires_grad_(True)
    bn = torch.nn.BatchNorm1d(64).to(device).train()
    def weight_grad():
        w.grad = None
        (h @ w).mul(grad).sum().backward()
        return w.grad.clone()
    def repeat(fn):
        first = fn()
        return max(float((first - fn()).abs().max()) for _ in range(3)), float(first.abs().max())
    reports = {
        "h.T @ grad (K=1e6, float32)": lambda: h.t() @ grad,
        "matmul weight gradient via autograd (K=1e6)": weight_grad,
        "BatchNorm1d training forward (N=1e6)": lambda: bn(h).detach(),
        "global sum (1e6, float32)": lambda: v.sum().reshape(1),
        "global mean/std (1e6, float32)": lambda: torch.stack([v.mean(), v.std()]),
        "row mean over 4096 x 244 (float32)": lambda: v[:4096 * 244].view(4096, 244).mean(1),
    }
    for name, fn in reports.items():
        diff, scale = repeat(fn)
        print(f"{name} on {device}: max |diff| over repeats {diff:.3e} (max |value| {scale:.3e}) -> "
              f"{'deterministic' if diff == 0 else 'ORDER-NONDETERMINISTIC'}")


def test_deterministic_weight_gradient_candidates_report(device):
    """Fixed-order alternatives for the MLP weight gradient on the device (cost and determinism)."""
    g = torch.Generator().manual_seed(21)
    n = 1_000_000
    h = torch.randn(n, 64, generator=g).to(device)
    grad = torch.randn(n, 32, generator=g).to(device)
    w = torch.randn(32, 64, generator=g).to(device)
    def chunked_gemm(chunk):
        def fn():
            acc = torch.zeros(32, 64, device=device)
            for start in range(0, n, chunk):
                acc = acc + grad[start:start + chunk].t() @ h[start:start + chunk]
            return acc
        return fn
    def outer_sum(chunk):
        def fn():
            acc = torch.zeros(32, 64, device=device)
            for start in range(0, n, chunk):
                acc = acc + (grad[start:start + chunk, :, None] * h[start:start + chunk, None, :]).sum(0)
            return acc
        return fn
    def timed(fn):
        accelerator.synchronize(); fn(); accelerator.synchronize()
        started = time.time(); first = fn(); accelerator.synchronize(); elapsed = time.time() - started
        diff = max(float((first - fn()).abs().max()) for _ in range(3))
        return diff, elapsed * 1e3
    reports = {
        "plain grad.T @ h (K=1e6)": lambda: grad.t() @ h,
        "chunked GEMM K=4096": chunked_gemm(4096), "chunked GEMM K=16384": chunked_gemm(16384),
        "chunked GEMM K=65536": chunked_gemm(65536), "outer-product sum chunk 8192": outer_sum(8192),
        "outer-product sum chunk 32768": outer_sum(32768),
        "forward x @ W.t (K=64)": lambda: h @ w.t(), "input gradient grad @ W (K=32)": lambda: grad @ w,
    }
    for name, fn in reports.items():
        diff, ms = timed(fn)
        print(f"{name} on {device}: max |diff| {diff:.3e}, {ms:.1f} ms -> {'deterministic' if diff == 0 else 'NONDETERMINISTIC'}")


def test_deterministic_weight_gradient_bmm_report(device):
    """Batched small-K partial GEMMs + fixed-order sum, and fp64 GEMM, as deterministic weight-gradient paths."""
    g = torch.Generator().manual_seed(23)
    n = 1_000_000
    h = torch.randn(n, 64, generator=g).to(device)
    grad = torch.randn(n, 32, generator=g).to(device)
    def bmm_partials(block):
        def fn():
            b = n // block
            partial = torch.bmm(grad[:b * block].view(b, block, 32).transpose(1, 2), h[:b * block].view(b, block, 64))
            return partial.sum(0)
        return fn
    def bmm_tree(block):
        def fn():
            b = n // block
            partial = torch.bmm(grad[:b * block].view(b, block, 32).transpose(1, 2), h[:b * block].view(b, block, 64))
            while partial.shape[0] > 1:
                if partial.shape[0] % 2:
                    partial = torch.cat([partial, torch.zeros_like(partial[:1])])
                partial = partial[0::2] + partial[1::2]
            return partial[0]
        return fn
    def timed(fn):
        accelerator.synchronize(); fn(); accelerator.synchronize()
        started = time.time(); first = fn(); accelerator.synchronize(); elapsed = time.time() - started
        diff = max(float((first - fn()).abs().max()) for _ in range(3))
        return diff, elapsed * 1e3, first
    reference = (grad.double().t() @ h.double())
    reports = {"bmm blocks of 64 + sum(0)": bmm_partials(64), "bmm blocks of 256 + sum(0)": bmm_partials(256),
               "bmm blocks of 1024 + sum(0)": bmm_partials(1024), "bmm blocks of 256 + pairwise tree": bmm_tree(256),
               "fp64 GEMM grad.T @ h": lambda: grad.double().t() @ h.double()}
    for name, fn in reports.items():
        diff, ms, value = timed(fn)
        err = float((value.double() - reference).abs().max() / reference.abs().max())
        print(f"{name} on {device}: max |diff| {diff:.3e}, {ms:.1f} ms, rel err vs fp64 {err:.2e} -> "
              f"{'deterministic' if diff == 0 else 'NONDETERMINISTIC'}")
    try:
        torch.use_deterministic_algorithms(True)
        diff, ms, _ = timed(lambda: grad.t() @ h)
        print(f"plain GEMM under use_deterministic_algorithms(True): max |diff| {diff:.3e}, {ms:.1f} ms")
    except Exception as exc:  # noqa: BLE001
        print(f"use_deterministic_algorithms(True): {type(exc).__name__}: {str(exc)[:120]}")
    finally:
        torch.use_deterministic_algorithms(False)
