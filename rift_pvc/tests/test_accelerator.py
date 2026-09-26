"""Tests for rift_pvc.accelerator; device parts skip when no accelerator is present."""
import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rift_pvc import accelerator as acc  # noqa: E402

needs_device = pytest.mark.skipif(not acc.is_available(), reason="no CUDA/XPU device in this process")


def test_backend_is_consistent():
    b = acc.backend()
    assert b in ("cuda", "xpu", "cpu")
    assert acc.is_available() == (b != "cpu")
    assert acc.device().type == b
    assert acc.device(0).type == b
    assert (acc.module() is None) == (b == "cpu")
    assert acc.rng_state_key() == f"torch_{b}_all"


def test_cuda_key_matches_train_py_checkpoint_key():
    # train.py stores the CUDA generator states under this exact key.
    if acc.backend() == "cuda":
        assert acc.rng_state_key() == "torch_cuda_all"


def test_env_override_to_cpu(monkeypatch):
    monkeypatch.setenv("RIFT_ACCELERATOR", "cpu")
    assert acc.backend() == "cpu"
    assert acc.device(3) == torch.device("cpu")
    assert acc.device_count() == 0
    assert acc.get_rng_state_all() == []
    assert acc.mem_get_info() == (0, 0)
    assert acc.collective_backend() == "gloo"
    assert acc.get_device_name() == "cpu"
    acc.manual_seed_all(1)  # no-op
    acc.synchronize()
    acc.empty_cache()


def test_env_override_rejects_unknown_and_unavailable(monkeypatch):
    monkeypatch.setenv("RIFT_ACCELERATOR", "tpu")
    with pytest.raises(ValueError):
        acc.backend()
    if not torch.cuda.is_available():
        monkeypatch.setenv("RIFT_ACCELERATOR", "cuda")
        with pytest.raises(RuntimeError):
            acc.backend()


def test_is_accelerator_predicate():
    assert acc.is_accelerator("cuda:0") and acc.is_accelerator(torch.device("xpu"))
    assert not acc.is_accelerator("cpu")


def test_collective_backend_value():
    assert acc.collective_backend() in ("nccl", "xccl", "gloo")
    if acc.backend() == "cuda":
        assert acc.collective_backend() == "nccl"


def test_autocast_context_runs():
    dtype = torch.bfloat16 if acc.backend() == "cpu" else torch.float16
    with acc.autocast(dtype=dtype):
        y = torch.randn(64, 64, device=acc.device()) @ torch.randn(64, 64, device=acc.device())
    assert y.dtype == dtype


def test_describe_keys():
    d = acc.describe()
    assert {"backend", "torch", "device_count", "devices"} <= set(d)
    assert d["device_count"] == len(d["devices"])


@needs_device
def test_device_rng_roundtrip():
    acc.manual_seed_all(7)
    saved = acc.get_rng_state_all()
    assert len(saved) == acc.device_count() and all(s.dtype == torch.uint8 for s in saved)
    x1 = torch.rand(16, device=acc.device())
    acc.set_rng_state_all(saved)
    x2 = torch.rand(16, device=acc.device())
    assert torch.equal(x1, x2)


@needs_device
def test_memory_api():
    acc.reset_peak_memory_stats()
    keep = torch.empty(1 << 24, dtype=torch.uint8, device=acc.device())
    acc.synchronize()
    assert acc.max_memory_allocated() >= keep.numel()
    assert acc.memory_allocated() >= keep.numel()
    free, total = acc.mem_get_info()
    assert 0 < free <= total and acc.total_memory() == total or acc.total_memory() > 0
    assert isinstance(acc.get_device_name(0), str) and acc.get_device_name(0)
    del keep
    acc.empty_cache()


@needs_device
def test_fp64_and_complex_on_device():
    a = torch.randn(128, 128, dtype=torch.float64)
    assert (a.to(acc.device()) @ a.to(acc.device()) - (a @ a).to(acc.device())).abs().max().item() < 1e-9
    z = torch.randn(4096, dtype=torch.complex128)
    err = (torch.fft.fft(z.to(acc.device())).cpu() - torch.fft.fft(z)).norm() / z.norm()
    assert err.item() < 1e-13
