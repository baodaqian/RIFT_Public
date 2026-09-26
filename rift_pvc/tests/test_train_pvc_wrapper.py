"""train_pvc.py rebinds only the accelerator-specific names of the unchanged train module."""
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train as _train  # noqa: E402
import train_pvc  # noqa: E402
from rift_pvc import accelerator, distributed as distributed_pvc, train_rng  # noqa: E402


def test_install_rebinds_exactly_the_five_names():
    before = {name: getattr(_train, name) for name in train_pvc.REBIND}
    train_pvc.install()
    assert _train.init_distributed is distributed_pvc.init_distributed
    assert _train.all_reduce_int is distributed_pvc.all_reduce_int
    assert _train.set_seed is train_rng.set_seed
    assert _train.capture_rng_state is train_rng.capture_rng_state
    assert _train.restore_rng_state is train_rng.restore_rng_state
    # the originals still exist in their own modules, untouched
    import rift.distributed as rd
    assert rd.init_distributed is not distributed_pvc.init_distributed
    assert all(callable(v) for v in before.values())


def test_train_file_on_disk_is_not_modified():
    src = (ROOT / "train.py").read_text()
    assert "rift_pvc" not in src and "torch.xpu" not in src


def test_backend_gate_refuses_non_xpu_without_override(monkeypatch):
    if accelerator.backend() == "xpu":
        pytest.skip("gate only observable without an XPU")
    monkeypatch.delenv("RIFT_PVC_ALLOW_BACKEND", raising=False)
    with pytest.raises(RuntimeError):
        train_pvc.check_backend()
    monkeypatch.setenv("RIFT_PVC_ALLOW_BACKEND", accelerator.backend())
    assert train_pvc.check_backend() == accelerator.backend()


def test_entry_point_help_runs_in_subprocess():
    env = dict(os.environ, RIFT_PVC_ALLOW_BACKEND="cpu,cuda,xpu", PYTHONPATH=str(ROOT))
    proc = subprocess.run([sys.executable, str(ROOT / "train_pvc.py"), "--help"], cwd=ROOT,
                          env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "--adaptive-capacity-v2" in proc.stdout or "--scene-repr" in proc.stdout


def test_rng_twins_roundtrip_and_reuse_train_helpers():
    train_rng.set_seed(11)
    state = train_rng.capture_rng_state()
    assert {"python", "numpy", "torch_cpu", "freq_cpu"} <= set(state)
    if accelerator.is_available():
        assert accelerator.rng_state_key() in state and state["accelerator_backend"] == accelerator.backend()
    a = torch.rand(4); fa = torch.randperm(10, generator=_train._FREQ_RNG)
    assert train_rng.restore_rng_state(state, require_complete=accelerator.is_available()) is True
    b = torch.rand(4); fb = torch.randperm(10, generator=_train._FREQ_RNG)
    assert torch.equal(a, b) and torch.equal(fa, fb)


def test_foreign_backend_payload_is_refused_under_strict_contract():
    state = train_rng.capture_rng_state()
    other = "torch_cuda_all" if accelerator.backend() != "cuda" else "torch_xpu_all"
    state = {k: v for k, v in state.items() if k != accelerator.rng_state_key()}
    state[other] = [torch.zeros(16, dtype=torch.uint8)]
    state["accelerator_backend"] = other.split("_")[1]
    with pytest.raises(ValueError):
        train_rng.restore_rng_state(state, require_complete=True)
    assert train_rng.restore_rng_state(state, require_complete=False) is False


@pytest.mark.skipif(not accelerator.is_available(), reason="needs a device")
def test_init_distributed_single_process_binds_device():
    rank, world, device = distributed_pvc.init_distributed()
    assert (rank, world) == (0, 1)
    assert torch.device(device).type == accelerator.backend()
    x = torch.ones(3, device=device)
    assert distributed_pvc.all_reduce_int(5) == 5 and x.sum().item() == 3
