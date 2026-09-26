"""GeRaF PVC adaptation gates that run without a device (Package B).

Every check here is an API, behaviour or execution check. Per ``AGENTS.md`` no
content-hash verification is performed on the vendored copies; their provenance
is the upstream URL and commit recorded in
``rift_pvc/vendor/geraf_sens/NOTICE.md``.
"""
from __future__ import annotations

import ast
import copy
import inspect
import re
from pathlib import Path

import numpy as np
import pytest
import torch

import train_geraf as cuda_entry
import train_geraf_pvc as pvc_entry
from rift import geraf_source_cli as cuda_cli
from rift import geraf_source_training as cuda_runtime
from rift_pvc import geraf_rng
from rift_pvc import geraf_source as source
from rift_pvc import geraf_source_cli as cli
from rift_pvc import geraf_source_training as runtime
from rift_pvc.geraf_autocast import autocast_fp16
from tests.test_geraf_source import SMALL, TinyData, compare_nested

ROOT = Path(__file__).resolve().parents[2]
PVC_GERAF_FILES = (
    "rift_pvc/geraf_source.py",
    "rift_pvc/geraf_source_training.py",
    "rift_pvc/geraf_source_cli.py",
    "rift_pvc/geraf_gotcha.py",
    "rift_pvc/geraf_rng.py",
    "rift_pvc/geraf_autocast.py",
    "rift_pvc/vendor/geraf_sens/rf_rendering.py",
    "rift_pvc/vendor/geraf_sens/train_reference.py",
    "train_geraf_pvc.py",
)


@pytest.fixture(autouse=True)
def cpu_backend(monkeypatch):
    monkeypatch.setenv("RIFT_ACCELERATOR", "cpu")


def _code_lines(path: Path):
    """Source lines with comments and docstring bodies removed."""
    text = path.read_text()
    tree = ast.parse(text)
    doc_spans = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and body \
                and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            doc_spans.update(range(body[0].lineno, body[0].end_lineno + 1))
    return [(number, re.sub(r"#.*$", "", line))
            for number, line in enumerate(text.splitlines(), start=1) if number not in doc_spans]


# --------------------------------------------------------------------------
# Adaptation surface
# --------------------------------------------------------------------------

def test_no_cuda_api_call_survives_in_the_pvc_geraf_files():
    offenders = []
    for name in PVC_GERAF_FILES:
        for number, line in _code_lines(ROOT / name):
            if "torch.cuda." in line and "torch.cuda.is_available()" not in line \
                    and "torch.cuda.set_rng_state_all" not in line \
                    and "torch.cuda.get_rng_state_all" not in line:
                offenders.append(f"{name}:{number}: {line.strip()}")
            if re.search(r"autocast\(\s*[\"']cuda[\"']", line):
                offenders.append(f"{name}:{number}: {line.strip()}")
    assert offenders == []


def test_the_only_remaining_cuda_mentions_are_the_audited_ones():
    """CUDA survives only in per-type device gates and the backend-keyed RNG
    payload, and the inert vendored backend strings stay verbatim."""
    assert [line.strip() for _, line in _code_lines(ROOT / "rift_pvc/geraf_source_training.py")
            if "cuda" in line] == [
        "if device.type == 'cuda' and not torch.cuda.is_available():",
        "if device.type == 'cuda' and saved.get('rng_cuda'):",
        "torch.cuda.set_rng_state_all(saved['rng_cuda'])",
        "rng_torch=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],",
    ]
    assert [line.strip() for _, line in _code_lines(ROOT / "rift_pvc/vendor/geraf_sens/rf_rendering.py")
            if "cuda" in line] == ['rt_backend: str = "cuda",', 'mf_backend: str = "cuda",'] * 2


def test_vendored_backend_strings_are_inert_and_never_select_a_kernel():
    """Why the vendored ``"cuda"`` defaults could be left exactly as upstream."""
    from rift import geraf_source_ops as ops
    for function in (ops.lensless_ray_tracer, ops.base_matched_filter):
        assert "backend" in inspect.signature(function).parameters
        body = inspect.getsource(function)
        assert "radar_cfg['native']" in body
        # The keyword is accepted and never read after the signature line.
        assert not re.search(r"\bbackend\b", body.split("):", 1)[1])
    model = source.build_model(source.recipe_from_config(SMALL, 0.15))
    assert (model.rt_backend, model.mf_backend) == ("native", "native")


def test_no_device_side_generator_anywhere_in_the_geraf_paths():
    """GeRaF's only randomness is numpy on the host, so XPU keeps the CUDA stream."""
    for name in PVC_GERAF_FILES:
        text = (ROOT / name).read_text()
        assert "torch.Generator(" not in text, name
        assert "torch.randperm" not in text, name
    sample = (ROOT / "rift/vendor/geraf_sens/sample.py").read_text()
    assert "np.random.rand" in sample and "torch.rand" not in sample


def test_autocast_twin_is_inert_on_cpu_exactly_like_the_cuda_original():
    a, b = torch.randn(4, 4), torch.randn(4, 4)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        original_dtype = (a @ b).dtype
    with autocast_fp16():
        assert (a @ b).dtype == original_dtype == torch.float32
    # A naive torch.autocast("cpu", float16) would NOT be a faithful twin.
    with torch.autocast(device_type="cpu", dtype=torch.float16):
        assert (a @ b).dtype == torch.float16


# --------------------------------------------------------------------------
# CLI and entry point
# --------------------------------------------------------------------------

def _argv(tmp_path, extra=()):
    npz, manifest = tmp_path / "b787.npz", tmp_path / "roles.json"
    npz.touch(), manifest.touch()
    return ["--implementation", "source_v1", "--npz-path", str(npz),
            "--role-manifest", str(manifest), "--checkpoint-dir", str(tmp_path / "ck"), *extra]


def test_source_cli_is_identical_to_the_original_except_the_device_default(tmp_path):
    argv = _argv(tmp_path)
    mine, theirs = vars(cli.parse_args(argv)), vars(cuda_cli.parse_args(argv))
    assert mine.pop("device") == "cpu"          # the active backend, via the shim
    assert theirs.pop("device") == "cuda"       # the CUDA original's literal
    assert mine == theirs
    assert cli.parse_args(_argv(tmp_path, ["--device", "xpu"])).device == "xpu"
    assert inspect.signature(cli.run) == inspect.signature(cuda_cli.run)


def test_entry_point_routes_source_v1_and_injects_a_device_on_the_legacy_path(tmp_path):
    args = pvc_entry.parse_args(_argv(tmp_path))
    assert args.implementation == "source_v1" and args.device == "cpu"
    # The compatibility parser's own default names neither the PVC card nor the
    # active backend, so the entry point injects the flag when it is omitted.
    legacy = pvc_entry.parse_args(["--implementation", "legacy", "--cache-root", str(tmp_path),
                                   "--checkpoint-dir", str(tmp_path / "ck"),
                                   "--npz-path", str(tmp_path / "b787.npz"),
                                   "--role-manifest", str(tmp_path / "roles.json")])
    assert legacy.device == "cpu"
    explicit = pvc_entry.parse_args(["--implementation", "legacy", "--cache-root", str(tmp_path),
                                     "--checkpoint-dir", str(tmp_path / "ck"), "--device", "xpu",
                                     "--npz-path", str(tmp_path / "b787.npz"),
                                     "--role-manifest", str(tmp_path / "roles.json")])
    assert explicit.device == "xpu"


def test_entry_point_exit_convention_matches_the_original():
    """A source_v1 run returns the trainer's result dict. Wrapping ``main()`` in
    ``SystemExit`` would make a successful run exit 1 - observed in smoke job
    2153319, where the target-cache stage reported ``status: prepared`` and the
    process still exited 1, cancelling its dependent job."""
    for name in ("train_geraf.py", "train_geraf_pvc.py"):
        guard = [node for node in ast.parse((ROOT / name).read_text()).body
                 if isinstance(node, ast.If)][-1]
        assert isinstance(guard.test, ast.Compare)          # if __name__ == "__main__"
        assert not any(isinstance(node, ast.Raise) for node in ast.walk(guard)), name
        assert [node.func.id for node in ast.walk(guard)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)] == ["main"], name
    # The only nonzero status is the clean-interruption code, raised by run().
    assert "SystemExit(143)" in (ROOT / "rift_pvc/geraf_source_cli.py").read_text()


def test_entry_point_refuses_a_foreign_backend_unless_overridden(monkeypatch):
    monkeypatch.delenv("RIFT_PVC_ALLOW_BACKEND", raising=False)
    with pytest.raises(RuntimeError, match="PVC entry point"):
        pvc_entry.check_backend()
    monkeypatch.setenv("RIFT_PVC_ALLOW_BACKEND", "cpu")
    assert pvc_entry.check_backend() == "cpu"


def test_installing_the_rng_twins_never_touches_the_cuda_module_by_default():
    """Importing the PVC entry point must not rebind train_geraf's helpers."""
    assert cuda_entry._seed_everything.__module__ == "train_geraf"
    assert cuda_entry._capture_rng_state.__module__ == "train_geraf"
    originals = {name: getattr(cuda_entry, name) for name in pvc_entry.REBIND}
    try:
        pvc_entry.install()
        assert cuda_entry._capture_rng_state.__module__ == "rift_pvc.geraf_rng"
        assert cuda_entry._restore_rng_state.__module__ == "rift_pvc.geraf_rng"
    finally:
        for name, function in originals.items():
            setattr(cuda_entry, name, function)


# --------------------------------------------------------------------------
# RNG twins
# --------------------------------------------------------------------------

class _Sampler:
    def __init__(self, value=0):
        self.value = value

    def state_dict(self):
        return {"cursor": self.value}

    def load_state_dict(self, state):
        self.value = state["cursor"]


def test_rng_twin_carries_both_payloads_and_round_trips():
    geraf_rng.seed_everything(42)
    state = geraf_rng.capture_rng_state(_Sampler(7))
    assert set(state) == {"python", "numpy_global", "torch_cpu", "torch_cuda",
                          "torch_xpu", "accelerator_backend", "view_sampler"}
    # The original key and its None-off-CUDA convention are preserved.
    original = cuda_entry._capture_rng_state(_Sampler(7))
    assert state["torch_cuda"] == original["torch_cuda"] is None
    assert state["torch_xpu"] is None and state["accelerator_backend"] == "cpu"
    first = torch.rand(4)
    sampler = _Sampler(0)
    geraf_rng.restore_rng_state(state, sampler)
    assert sampler.value == 7
    torch.testing.assert_close(torch.rand(4), first, rtol=0, atol=0)


def test_a_cuda_written_payload_restores_on_a_non_cuda_host():
    """The tolerance the original already has, kept: no accelerator payload is
    required, and only the host streams are restored."""
    geraf_rng.seed_everything(11)
    foreign = dict(cuda_entry._capture_rng_state(_Sampler(3)),
                   torch_cuda=[torch.zeros(16, dtype=torch.uint8)])
    sampler = _Sampler(0)
    geraf_rng.restore_rng_state(foreign, sampler)   # must not raise
    assert sampler.value == 3


# --------------------------------------------------------------------------
# Execution: the PVC lifecycle must equal the CUDA module's on the same host
# --------------------------------------------------------------------------

def test_pvc_and_cuda_training_lifecycles_agree_bitwise_on_cpu(tmp_path):
    data_a, data_b = TinyData(), TinyData()
    mine = runtime.train(data=data_a, output_dir=tmp_path / "pvc", config=SMALL, device="cpu")
    theirs = cuda_runtime.train(data=data_b, output_dir=tmp_path / "cuda", config=SMALL, device="cpu")
    assert mine["status"] == theirs["status"] == "complete"
    a = torch.load(tmp_path / "pvc" / "checkpoint_latest.pth.tar", weights_only=False)
    b = torch.load(tmp_path / "cuda" / "checkpoint_latest.pth.tar", weights_only=False)
    compare_nested(a["models"], b["models"])
    compare_nested(a["optimizer"], b["optimizer"])
    assert a["validation_history"] == b["validation_history"]
    assert a["recipe"] == b["recipe"] and a["complete"] and a["target_identity"] == b["target_identity"]
    assert data_a.reads == data_b.reads


def test_checkpoint_keeps_rng_cuda_and_adds_rng_xpu(tmp_path):
    data = TinyData()
    runtime.train(data=data, output_dir=tmp_path / "run", config=SMALL, device="cpu")
    saved = torch.load(tmp_path / "run" / "checkpoint_latest.pth.tar", weights_only=False)
    assert saved["rng_cuda"] == [] and saved["rng_xpu"] == []
    # The extra key must not disturb the unchanged checkpoint contract.
    recipe = source.recipe_from_config(SMALL, data.extent)
    runtime.validate_checkpoint(saved, data, recipe)
    cuda_runtime.validate_checkpoint(saved, data, recipe)


def test_a_cuda_checkpoint_without_rng_xpu_resumes_under_the_pvc_runtime(tmp_path):
    """Cross-backend resume stays as tolerated as the original makes it."""
    import signal as signal_module
    data = TinyData()
    original_step = torch.optim.AdamW.step
    calls = [0]

    def stop_after_first(optimizer, *args, **kwargs):
        result = original_step(optimizer, *args, **kwargs)
        calls[0] += 1
        if calls[0] == 1:
            signal_module.getsignal(signal_module.SIGTERM)(signal_module.SIGTERM, None)
        return result

    out = tmp_path / "run"
    with pytest.MonkeyPatch.context() as m:
        m.setattr(torch.optim.AdamW, "step", stop_after_first)
        assert cuda_runtime.train(data=data, output_dir=out, config=SMALL,
                                  device="cpu")["status"] == "interrupted"
    stopped = torch.load(out / "checkpoint_latest.pth.tar", weights_only=False)
    assert "rng_xpu" not in stopped          # written by the CUDA module
    resumed = runtime.train(data=data, output_dir=out, config=SMALL, device="cpu", resume="auto")
    assert resumed["status"] == "complete"


def test_train_device_default_resolves_through_the_shim(tmp_path):
    parameters = inspect.signature(runtime.train).parameters
    assert list(parameters) == list(inspect.signature(cuda_runtime.train).parameters)
    assert parameters["device"].default is None
    assert inspect.signature(cuda_runtime.train).parameters["device"].default == "cuda"
    # Omitting the device must land on the active backend, not on a literal.
    assert runtime.train(data=TinyData(), output_dir=tmp_path / "default",
                         config=SMALL)["status"] == "complete"


def test_an_unavailable_requested_device_is_refused_by_type(tmp_path):
    for name, message in (("cuda", "CUDA requested"), ("xpu", "XPU requested")):
        if torch.device(name).type == "cuda" and torch.cuda.is_available():
            continue
        if name == "xpu" and torch.xpu.is_available():
            continue
        with pytest.raises(RuntimeError, match=message):
            runtime.train(data=TinyData(), output_dir=tmp_path / name, config=SMALL, device=name)


def test_gotcha_binding_forwards_to_the_pvc_runtime(monkeypatch):
    from rift_pvc import geraf_gotcha
    from rift import geraf_gotcha as cuda_gotcha
    assert inspect.signature(geraf_gotcha.run_gotcha) == inspect.signature(cuda_gotcha.run_gotcha)
    assert geraf_gotcha.train is runtime.train
    seen = {}
    monkeypatch.setattr(geraf_gotcha, "train", lambda **kwargs: seen.update(kwargs) or "ok")
    monkeypatch.setattr(geraf_gotcha, "GOTCHASourceData", lambda dataset: f"data({dataset})")
    assert geraf_gotcha.run_gotcha(dataset="d", output_dir="o", config={}, device="cpu", resume=None) == "ok"
    assert seen["data"] == "data(d)" and seen["device"] == "cpu"
