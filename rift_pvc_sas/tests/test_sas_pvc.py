"""Package G (RIFT-SAS, adaptive RIFT-SAS, SH-SAS on sonar caches): gates that run without a device.

Adaptation surface (which CUDA touches survive and where), the CLI and
checkpoint contracts of ``train_sas_pvc.py`` against the unchanged
``train_sas.py``, the structural copy of ``main``, exact interrupt/resume of
all three models on a synthetic cache, the validation-only read-out's sealed
test role, and the PVC comparison launcher's argument list.

Run with ``RIFT_ACCELERATOR=cpu``; the fixture sets it and allows the CPU
backend for the entry point."""
from __future__ import annotations

import ast
import difflib
import inspect
import json
import os
import re
import signal
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_sas as cuda_entry  # noqa: E402
import train_sas_pvc as pvc_entry  # noqa: E402
from rift.sas_dataset import load_sas_cache  # noqa: E402
from rift_pvc import accelerator  # noqa: E402
from rift_pvc_sas import synthetic as sas_synthetic  # noqa: E402
from rift_pvc_sas import training as twins  # noqa: E402

CUDA_SAS_FILES = ("train_sas.py", "rift/rift_sas.py", "rift/sas_operator.py", "rift/sas_dataset.py",
                  "rift/airsas_contract.py", "rift/sh_sas.py", "rift/sparse_scene.py",
                  "scripts/run_airsas_comparison.sh", "scripts/prepare_airsas_cache.py")
PVC_SAS_FILES = ("train_sas_pvc.py", "rift_pvc_sas/training.py", "rift_pvc_sas/synthetic.py")
PRODUCTION_FLAGS = ["--require-explicit-splits", "--sh-degree", "3", "--num-rays", "4900", "--max-bins", "110",
                    "--grad-clip", "1.0", "--beamwidth-deg", "30", "--sh-direction", "rx_to_point", "--seed", "42"]
MODELS = ("adaptive_rift_sas", "sh_sas", "rift_sas")


@pytest.fixture(autouse=True)
def cpu_backend(monkeypatch):
    monkeypatch.setenv("RIFT_ACCELERATOR", "cpu")
    monkeypatch.setenv("RIFT_PVC_ALLOW_BACKEND", "cpu")
    cuda_entry.STOP_REQUESTED = False
    yield
    cuda_entry.STOP_REQUESTED = False


@pytest.fixture(scope="module")
def cache_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic_cache")
    return sas_synthetic.write_synthetic_cache(root / "rings3", num_rings=3, ring_size=4, num_bins=8,
                                               grid_shape=(3, 3, 2), seed=0)


def small_args(model, cache, name, root, steps=8, extra=()):
    base = ["--cache", str(cache), "--model", model, "--checkpoint-root", str(root), "--checkpoint-name", name,
            "--require-explicit-splits", "--sh-degree", "3", "--num-rays", "16", "--max-bins", "4",
            "--grad-clip", "1.0", "--beamwidth-deg", "30", "--sh-direction", "rx_to_point", "--seed", "42",
            "--steps", str(steps), "--eval-every", "4", "--eval-pings", "1", "--eval-bins", "0",
            "--checkpoint-every", "2", "--log-every", "1", "--query-chunk", "4096", "--device", "cpu"]
    if model == "adaptive_rift_sas":
        base += ["--initial-granularity", "2", "--adaptive-capacity", "64", "--max-active", "64",
                 "--granularity", "4", "--refine-every", "4", "--probe-every", "2"]
    elif model == "sh_sas":
        base += ["--granularity", "4", "--hash-levels", "2", "--hash-base-resolution", "2",
                 "--hash-final-resolution", "4", "--hash-log2-size", "6", "--hidden-dim", "8"]
    else:
        base += ["--granularity", "4"]
    return base + list(extra)


class StopAfter:
    """Diagnostic observer that raises the trainer's own stop flag after one optimizer step."""

    def __init__(self, step):
        self.step = step

    def on_after_optimizer(self, step, **_):
        if step == self.step:
            cuda_entry.request_stop(signal.SIGTERM, None)


def _code_lines(path: Path):
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


def _tensor_tree_equal(a, b, path="", *, exact=True, rtol=0.0, atol=0.0):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor), path
        assert a.shape == b.shape and a.dtype == b.dtype, path
        if exact:
            assert torch.equal(a.cpu(), b.cpu()), f"tensor differs at {path}: max |diff| " \
                f"{(a.cpu().to(torch.float64 if not a.is_complex() else torch.complex128) - b.cpu().to(torch.float64 if not b.is_complex() else torch.complex128)).abs().max()}"
        else:
            torch.testing.assert_close(a.cpu(), b.cpu(), rtol=rtol, atol=atol, msg=path)
    elif isinstance(a, dict):
        assert set(a) == set(b), f"{path}: {set(a) ^ set(b)}"
        for key in a:
            _tensor_tree_equal(a[key], b[key], f"{path}.{key}", exact=exact, rtol=rtol, atol=atol)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _tensor_tree_equal(x, y, f"{path}[{i}]", exact=exact, rtol=rtol, atol=atol)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b), path
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def assert_payloads_match(resumed, reference, *, exact=True, rtol=1e-6, atol=1e-8):
    for key in ("sas_contract_version", "step", "epoch", "best_val_rel_mse", "model_kind", "calibration_mode",
                "rng_state", "shared_operator", "parameter_counts", "geometry_truth_used_for_training"):
        _tensor_tree_equal(resumed[key], reference[key], key)
    assert torch.equal(resumed["torch_rng_state"], reference["torch_rng_state"])
    strip = lambda rows: [{k: v for k, v in row.items() if k != "elapsed_seconds"} for row in rows]
    _tensor_tree_equal(strip(resumed["history"]), strip(reference["history"]), "history",
                       exact=exact, rtol=rtol, atol=atol)
    for key in ("model_state_dict", "calibration_state_dict", "optimizer_state_dict"):
        _tensor_tree_equal(resumed[key], reference[key], key, exact=exact, rtol=rtol, atol=atol)
    assert resumed["cache_manifest"] == reference["cache_manifest"]


# --------------------------------------------------------------------------
# Adaptation surface
# --------------------------------------------------------------------------

def test_cuda_files_on_disk_are_untouched():
    for name in CUDA_SAS_FILES:
        text = (ROOT / name).read_text()
        assert "rift_pvc" not in text and "xpu" not in text.lower(), name


def test_the_only_cuda_mentions_in_the_pvc_files_are_the_audited_guarded_ones():
    allowed = re.compile(r"torch\.cuda\.(is_available|manual_seed_all|set_rng_state_all|max_memory_allocated)\(")
    offenders = []
    for name in PVC_SAS_FILES:
        for number, line in _code_lines(ROOT / name):
            if "torch.cuda." in line and not allowed.search(line):
                offenders.append(f"{name}:{number}: {line.strip()}")
            if re.search(r"autocast\(\s*[\"']cuda[\"']", line) or ".cuda()" in line or "torch.Generator(device" in line:
                offenders.append(f"{name}:{number}: {line.strip()}")
    assert offenders == []


def test_install_rebinds_exactly_the_audited_names_and_is_idempotent():
    pvc_entry.install()
    pvc_entry.install()
    for name in twins.REBOUND:
        assert getattr(cuda_entry, name) is getattr(twins, name), name
        assert twins._ORIGINAL[name] is not getattr(twins, name)
    untouched = {n for n, v in vars(cuda_entry).items() if callable(v) and getattr(v, "__module__", "") == "train_sas"}
    assert untouched >= {"render_one", "evaluate", "build_model", "build_calibration", "_optimizer_for_model",
                         "_reconcile_saved_recipe", "_validate_cache_contract", "_resolve_selected_best", "main"}
    assert twins.REBOUND == ("parse_args", "seed_all", "checkpoint")


def test_main_copy_differs_from_the_original_only_in_the_audited_lines():
    original = inspect.getsource(cuda_entry.main).splitlines()
    copied = inspect.getsource(pvc_entry.main_copy).splitlines()
    diff = [line for line in difflib.unified_diff(original, copied, lineterm="", n=0)
            if line[:1] in "+-" and not line.startswith(("+++", "---"))]
    expected = [
        "-def main(",
        "+def main_copy(",
        '-        if torch.cuda.is_available() and state.get("cuda_rng_state") is not None:',
        '-            torch.cuda.set_rng_state_all(state["cuda_rng_state"])',
        "+        twins.restore_device_rng_state(state, device=device)",
        "-        if current % args.checkpoint_every == 0 or current == args.steps or STOP_REQUESTED:",
        "+        if current % args.checkpoint_every == 0 or current == args.steps or _sas.STOP_REQUESTED:",
        "-        if STOP_REQUESTED:",
        "+        if _sas.STOP_REQUESTED:",
        "+            **twins.readout_telemetry(device),",
    ]
    assert sorted(diff) == sorted(expected), "\n".join(diff)
    assert not re.search(r"(?<![._\w])STOP_REQUESTED", inspect.getsource(pvc_entry.main_copy))


# --------------------------------------------------------------------------
# CLI, payload and RNG twins
# --------------------------------------------------------------------------

def test_parse_args_matches_the_original_except_the_device_default():
    argv = ["--cache", "c", "--model", "adaptive_rift_sas", "--checkpoint-name", "n", *PRODUCTION_FLAGS, "--profile", "full"]
    theirs, mine = vars(cuda_entry.parse_args(argv)), vars(twins.parse_args(argv))
    assert mine.pop("device") == "cpu"          # accelerator device under RIFT_ACCELERATOR=cpu
    theirs.pop("device")
    assert mine == theirs
    assert twins.parse_args(argv + ["--device", "xpu"]).device == "xpu"
    assert twins.parse_args(argv + ["--device=cuda"]).device == "cuda"
    # the explicit-field scan of the original never sees the injected flag as a recipe field
    assert "device" not in cuda_entry._explicit_cli_fields(argv + ["--device", "xpu"])


def test_check_backend_refuses_cpu_without_the_override(monkeypatch):
    monkeypatch.delenv("RIFT_PVC_ALLOW_BACKEND")
    with pytest.raises(RuntimeError, match="PVC entry point"):
        twins.check_backend()
    monkeypatch.setenv("RIFT_PVC_ALLOW_BACKEND", "cpu")
    assert twins.check_backend() == "cpu"


def test_checkpoint_twin_adds_only_the_backend_keys(cache_root, tmp_path):
    cache = load_sas_cache(cache_root)
    args = twins.parse_args(small_args("rift_sas", cache_root, "payload", tmp_path))
    args.opacity_normalize = False
    device = torch.device("cpu")
    model = cuda_entry.build_model(args, cache, device)
    calibration = cuda_entry.build_calibration("legacy_cartesian", device)
    optimizer = cuda_entry._optimizer_for_model(model, calibration, args)
    rng = np.random.default_rng(0)
    original = twins._ORIGINAL["checkpoint"](model, calibration, optimizer, 3, 0.5, rng, [], args, cache)
    twin = twins.checkpoint(model, calibration, optimizer, 3, 0.5, rng, [], args, cache)
    assert set(twin) - set(original) == {"xpu_rng_state", "accelerator_backend"}
    assert twin["accelerator_backend"] == "cpu" and twin["xpu_rng_state"] is None and twin["cuda_rng_state"] is None
    for key in original:
        _tensor_tree_equal(twin[key], original[key], key)


def test_restore_device_rng_state_restores_the_active_backend_payload(monkeypatch, capsys):
    # CPU backend: nothing to restore, the notice names what the checkpoint carries
    assert twins.restore_device_rng_state({"cuda_rng_state": None, "xpu_rng_state": None}) is None
    assert "not restored" in capsys.readouterr().out
    # XPU backend (faked): the XPU payload is restored when its device count matches
    calls = []
    monkeypatch.setattr(accelerator, "backend", lambda: "xpu")
    monkeypatch.setattr(accelerator, "device_count", lambda: 1)
    monkeypatch.setattr(accelerator, "set_rng_state_all", lambda states: calls.append(list(states)))
    payload = [torch.zeros(16, dtype=torch.uint8)]
    assert twins.restore_device_rng_state({"xpu_rng_state": payload, "cuda_rng_state": None}) == "xpu_rng_state"
    assert calls == [payload]
    # a CUDA-written checkpoint on XPU: skipped with a notice, not refused
    assert twins.restore_device_rng_state({"cuda_rng_state": [torch.zeros(8, dtype=torch.uint8)],
                                           "accelerator_backend": "cuda"}) is None
    assert "written on 'cuda'" in capsys.readouterr().out
    # a payload for two devices on a one-device host is skipped, not applied
    assert twins.restore_device_rng_state({"xpu_rng_state": payload * 2}) is None
    assert calls == [payload]


def test_readout_telemetry_labels_the_backend():
    assert twins.readout_telemetry(torch.device("cpu")) == {"peak_accelerator_memory_bytes": 0,
                                                            "accelerator_backend": "cpu"}


# --------------------------------------------------------------------------
# Synthetic cache
# --------------------------------------------------------------------------

def test_synthetic_cache_loads_with_explicit_splits_and_is_not_an_airsas_identity(cache_root):
    cache = load_sas_cache(cache_root)
    assert cache.has_explicit_splits and cache.num_pings == 12 and cache.num_bins == 8
    assert cache.train_indices.tolist() == list(range(0, 4))
    assert cache.validation_indices.tolist() == list(range(4, 8))
    assert cache.test_indices.tolist() == list(range(8, 12))
    assert cache.tx_vecs is not None and cache.manifest["dataset_identity"] == sas_synthetic.IDENTITY
    from rift.airsas_contract import validate_cache_5k
    with pytest.raises(ValueError):
        validate_cache_5k(cache_root)
    assert sas_synthetic.ring_split(120, 360)["counts"] == {"train_rings": 96, "validation_rings": 12, "test_rings": 12}


# --------------------------------------------------------------------------
# Interrupt / resume through the entry point (all three models)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("model", MODELS)
def test_interrupted_fit_resumed_through_the_entry_point_equals_the_uninterrupted_run(model, cache_root, tmp_path):
    extra = ["--grid-shape", "2", "3", "2"] if model == "rift_sas" else []
    reference_argv = small_args(model, cache_root, "reference", tmp_path, extra=extra)
    pvc_entry.main(reference_argv)
    reference_dir = tmp_path / "reference"
    reference = torch.load(reference_dir / "checkpoint_final.pt", map_location="cpu", weights_only=False)
    assert reference["accelerator_backend"] == "cpu" and "xpu_rng_state" in reference
    assert json.loads((reference_dir / "status.json").read_text())["done"] is True
    readout = json.loads((reference_dir / "selected_readout.json").read_text())
    assert readout["accelerator_backend"] == "cpu" and readout["peak_cuda_memory_bytes"] == 0

    cuda_entry.STOP_REQUESTED = False
    interrupted_argv = small_args(model, cache_root, "interrupted", tmp_path, extra=extra)
    pvc_entry.main(interrupted_argv, diagnostic_observer=StopAfter(3))
    run_dir = tmp_path / "interrupted"
    status = json.loads((run_dir / "status.json").read_text())
    assert status == {"done": False, "step": 3, "reason": "signal", "model": model}
    latest = torch.load(run_dir / "checkpoint_latest.pt", map_location="cpu", weights_only=False)
    assert latest["step"] == 3 and latest["accelerator_backend"] == "cpu"
    assert not (run_dir / "checkpoint_final.pt").exists()

    cuda_entry.STOP_REQUESTED = False
    pvc_entry.main(interrupted_argv + ["--resume", str(run_dir / "checkpoint_latest.pt")])
    resumed = torch.load(run_dir / "checkpoint_final.pt", map_location="cpu", weights_only=False)
    assert json.loads((run_dir / "status.json").read_text())["done"] is True
    assert_payloads_match(resumed, reference, exact=True)
    if model == "adaptive_rift_sas":
        assert resumed["parameter_counts"]["active_points"] >= 8   # the step-4 refinement ran after the resume
    assert (run_dir / "density.npy").exists() and (run_dir / "history.csv").exists()


def test_recipe_mismatch_on_resume_is_still_refused(cache_root, tmp_path):
    argv = small_args("rift_sas", cache_root, "recipe", tmp_path, steps=2)
    pvc_entry.main(argv)
    cuda_entry.STOP_REQUESTED = False
    with pytest.raises(ValueError, match="checkpoint recipe mismatch"):
        pvc_entry.main(argv[:-2] + ["--device", "cpu", "--opacity-scale", "7",
                                    "--resume", str(tmp_path / "recipe" / "checkpoint_final.pt")])


# --------------------------------------------------------------------------
# Validation-only read-out keeps the test role sealed
# --------------------------------------------------------------------------

class _RecordingWeights:
    def __init__(self, weights):
        self._weights = weights
        self.pings = []

    @property
    def shape(self):
        return self._weights.shape

    def __getitem__(self, item):
        ping = item[0] if isinstance(item, tuple) else item
        self.pings.append(int(ping) if np.ndim(ping) == 0 else np.asarray(ping).tolist())
        return self._weights[item]


def test_validation_readout_reads_no_test_ping(cache_root, tmp_path, monkeypatch):
    argv = small_args("sh_sas", cache_root, "readout", tmp_path, steps=4)
    pvc_entry.main(argv)
    cache = load_sas_cache(cache_root)
    recorder = _RecordingWeights(cache.weights)
    watched = cache.__class__(**{**cache.__dict__, "weights": recorder})
    monkeypatch.setattr(pvc_entry, "load_sas_cache", lambda *_a, **_k: watched)   # the copied main binds it
    cuda_entry.STOP_REQUESTED = False
    pvc_entry.main(argv + ["--eval-only", "--evaluation-role", "validation", "--eval-pings", "0",
                           "--resume", str(tmp_path / "readout" / "checkpoint_best.pt")])
    touched = {p for entry in recorder.pings for p in (entry if isinstance(entry, list) else [entry])}
    assert touched and touched <= set(cache.validation_indices.tolist())
    assert not touched & set(cache.test_indices.tolist())
    report = json.loads((tmp_path / "readout" / "validation_eval.json").read_text())
    assert report["evaluation_role"] == "validation" and report["views"] == 4.0


# --------------------------------------------------------------------------
# The PVC comparison launcher
# --------------------------------------------------------------------------

def _stubbed_launch(tmp_path, env):
    stub_dir = tmp_path / "stub"
    stub_dir.mkdir(exist_ok=True)
    log = tmp_path / "calls.log"
    stub = stub_dir / "python"
    stub.write_text("#!/bin/bash\nprintf '%s\\n' \"$@\" >> \"$STUB_LOG\"\necho --- >> \"$STUB_LOG\"\n")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    full_env = {**os.environ, **env, "PATH": f"{stub_dir}:{os.environ['PATH']}", "STUB_LOG": str(log)}
    result = subprocess.run(["bash", str(ROOT / "scripts_pvc_sas" / "run_airsas_comparison_pvc.sh")],
                            env=full_env, capture_output=True, text=True)
    calls = [block.splitlines() for block in log.read_text().split("---\n") if block.strip()] if log.exists() else []
    return result, calls


def test_comparison_launcher_renders_the_production_argument_list_for_the_pvc_entry(tmp_path):
    env = {"SCENE": "armadillo", "MODEL": "adaptive_rift_sas", "CACHE": str(tmp_path / "cache"), "PROFILE": "full",
           "CHECKPOINT_ROOT": str(tmp_path / "ck"), "CHECKPOINT_NAME": "airsas_armadillo5k_adaptive_rift_sas_full_v1_pvcsmoke"}
    result, calls = _stubbed_launch(tmp_path, env)
    assert result.returncode == 0, result.stderr
    assert len(calls) == 2
    assert calls[0][0].endswith("scripts/validate_airsas_5k_cache.py") and calls[0][1] == env["CACHE"]
    assert calls[1][0] == str(ROOT / "train_sas_pvc.py")
    assert calls[1][1:] == ["--cache", env["CACHE"], "--model", "adaptive_rift_sas", "--checkpoint-root", env["CHECKPOINT_ROOT"],
                            "--checkpoint-name", env["CHECKPOINT_NAME"], *PRODUCTION_FLAGS, "--profile", "full"]
    # the CUDA launcher renders the same list for train_sas.py
    original = (ROOT / "scripts/run_airsas_comparison.sh").read_text()
    pvc = (ROOT / "scripts_pvc_sas/run_airsas_comparison_pvc.sh").read_text()
    assert original.replace('exec python "$PROJECT/train_sas.py" "${ARGS[@]}"', "") in pvc.replace(
        'exec python "$PROJECT/train_sas_pvc.py" "${ARGS[@]}"', "").replace(
        'if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then\n  [[ -s "$RESUME_CHECKPOINT" ]] || { echo "RESUME_CHECKPOINT does not exist: $RESUME_CHECKPOINT" >&2; exit 2; }\n  ARGS+=(--resume "$RESUME_CHECKPOINT")\nfi\n', "").replace(
        "# PVC twin of scripts/run_airsas_comparison.sh (Package G): identical environment\n"
        "# contract and argument list; only the trainer is train_sas_pvc.py. One addition:\n"
        "# RESUME_CHECKPOINT=<path> appends --resume <path> (explicit resume for the smokes).\n", "")


def test_comparison_launcher_resume_and_rejections(tmp_path):
    base = {"SCENE": "armadillo", "MODEL": "sh_sas", "CACHE": str(tmp_path / "cache"), "PROFILE": "smoke",
            "CHECKPOINT_ROOT": str(tmp_path / "ck")}
    ckpt = tmp_path / "checkpoint_latest.pt"
    ckpt.write_bytes(b"x")
    result, calls = _stubbed_launch(tmp_path, {**base, "RESUME_CHECKPOINT": str(ckpt)})
    assert result.returncode == 0 and calls[1][-2:] == ["--resume", str(ckpt)]
    assert "--steps" in calls[1] and calls[1][calls[1].index("--steps") + 1] == "4"
    result, _ = _stubbed_launch(tmp_path, {**base, "RESUME_CHECKPOINT": str(tmp_path / "missing.pt")})
    assert result.returncode == 2
    result, _ = _stubbed_launch(tmp_path, {**base, "MODEL": "rift_sas"})
    assert result.returncode == 2
    result, _ = _stubbed_launch(tmp_path, {**base, "SCENE": "bunny"})
    assert result.returncode == 2
