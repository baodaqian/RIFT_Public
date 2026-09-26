"""Regression checks for the independent Package F audit findings."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from scripts_pvc.smoke_radarsplat_pvc import run_segment

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("body,match", [
    ("raise SystemExit(17)", "exit 17"),
    ("raise SystemExit(143)", "reason None"),
    ("print('Aten Op fallback from XPU to CPU', end='', flush=True)", "fallback"),
    ("import time; time.sleep(10)", "watchdog"),
])
def test_controller_rejects_errors_unrequested_signals_fallbacks_and_timeouts(tmp_path, body, match):
    with pytest.raises((RuntimeError, TimeoutError), match=match):
        run_segment([sys.executable, "-c", body], tmp_path/"child.log",
                    stop_step=100, deadline=.2, grace=2)


def test_controller_only_accepts_its_own_cooperative_stop(tmp_path):
    body = """import json,signal,time
signal.signal(signal.SIGTERM, lambda *_: exit(143))
print(json.dumps({'step': 100, 'total': 1.0}), flush=True)
while True: time.sleep(.01)
"""
    result = run_segment([sys.executable, "-c", body], tmp_path/"child.log",
                         stop_step=100, deadline=5, grace=2)
    assert result["returncode"] == 143 and result["stop_reason"] == "update_bound"


def test_missing_checkpoint_is_not_an_accepted_interruption(tmp_path):
    from scripts_pvc.smoke_radarsplat_pvc import load_partial
    with pytest.raises(RuntimeError, match="checkpoint is missing"):
        load_partial(tmp_path/"checkpoint_latest.pt", None)


@pytest.mark.parametrize("record", [None, {"schema": "wrong", "radarsplat_backend": "cuda"}])
def test_gotcha_checks_every_head_before_delegating(tmp_path, monkeypatch, record):
    from rift_pvc import radarsplat_gotcha as pvc
    folder = tmp_path/"vv/checkpoints"
    folder.mkdir(parents=True)
    (folder/"checkpoint_latest.pt").write_bytes(b"must reject before loading this checkpoint")
    if record is not None:
        (folder/"backend.json").write_text(json.dumps(record))
    monkeypatch.delenv("RIFT_PVC_RADARSPLAT_ALLOW_CUDA_RESUME", raising=False)
    monkeypatch.setattr(pvc._backend, "run_gotcha", lambda **_: pytest.fail("read/prepared another head before backend check"))
    with pytest.raises(RuntimeError, match="no backend.json|not.*fork_torch_mirror"):
        pvc.run_gotcha(dataset=SimpleNamespace(polarizations=["hh", "vv"]), output_dir=tmp_path,
                       config={}, device="cpu", resume=tmp_path/"radarsplat_gotcha.pt")


def test_completed_gotcha_head_cannot_bypass_sidecar(tmp_path, monkeypatch):
    from tests.radarsplat_gotcha_fixtures import write_shard, tiny_region
    from tests.test_radarsplat_gotcha import CONFIG, test_shared_source_training_resume_and_gotcha_readout
    from rift.gotcha_dataset import GOTCHADataset
    write_shard(tmp_path/"data/New_Transfer/shards/pass1_hh.npz")
    dataset = GOTCHADataset(tmp_path/"data", passes=[1], region=tiny_region())
    test_shared_source_training_resume_and_gotcha_readout(dataset, tmp_path, monkeypatch, "budget48")
    folder = tmp_path/"run/hh/checkpoints"
    checkpoint = torch.load(folder/"checkpoint_latest.pt", map_location="cpu", weights_only=False)
    torch.save(checkpoint, folder/"checkpoint_final.pt")
    (folder/"backend.json").unlink(missing_ok=True)
    from rift_pvc import radarsplat_gotcha as pvc
    with pytest.raises(RuntimeError, match="no backend.json"):
        pvc.run_gotcha(dataset=dataset, output_dir=tmp_path/"run",
                       config={**CONFIG, "fidelity_profile": "budget48"}, device="cpu",
                       resume=tmp_path/"run"/pvc.CONTROL_FILE)


def test_collection_readout_cli_installs_pvc_in_fresh_process(tmp_path):
    # Exercise routing with a synthetic reader hook; a separate card check
    # reads the real B787 checkpoint through this same command line.
    code = """import sys
import scripts_pvc.readout_radarsplat_checkpoint_pvc as cli
import rift.radarsplat_release_training as engine
from rift_pvc.radarsplat_xpu_backend import load_xpu_reference
def reader(**kw):
    assert engine.load_cuda_reference is load_xpu_reference
    assert str(kw['device']) == 'cpu' and kw['role'] == 'validation'
    return {'pvc': {'readout_backend': 'fork_torch_mirror_xpu_v1'}}
cli.original_readout = reader
cli.main(['--checkpoint', 'dummy', '--cache-root', 'dummy', '--output', sys.argv[1]])
"""
    output = tmp_path/"readout.json"
    result = subprocess.run([sys.executable, "-c", code, str(output)], cwd=ROOT,
                            env={**os.environ, "PYTHONPATH": str(ROOT), "RIFT_ACCELERATOR": "cpu"},
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["pvc"]["readout_backend"] == "fork_torch_mirror_xpu_v1"
