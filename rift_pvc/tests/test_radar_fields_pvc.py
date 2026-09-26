"""Radar Fields PVC adaptation gates that run without a device (Package E).

Checks the adaptation surface (which CUDA touches survive and where), the CLI
contract of the ``_pvc`` entry point against the unchanged trainer, the
backend identity in the recipe contract and checkpoints, the release model
through the tinycudann torch shim on CPU, the native GOTCHA twin's
interrupt/resume path, and the frontend rewrites."""
from __future__ import annotations

import ast
import copy
import json
import os
import re
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
for entry in (ROOT, ROOT / "tests"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import train_radar_fields as cuda_entry  # noqa: E402
import train_radar_fields_pvc as pvc_entry  # noqa: E402
import train_rift_dataset_pvc as fe  # noqa: E402
import train_gotcha_dataset_pvc as gotcha_fe  # noqa: E402
from train_rift_dataset import commands_for  # noqa: E402
from rift import radar_fields_recipe  # noqa: E402
from rift import radar_fields_upstream as cuda_upstream  # noqa: E402
from rift import radar_fields_gotcha as cuda_gotcha  # noqa: E402
from rift_pvc import radar_fields_gotcha as pvc_gotcha  # noqa: E402
from rift_pvc import radar_fields_training as twins  # noqa: E402
from rift_pvc import radar_fields_upstream as pvc_upstream  # noqa: E402
from rift_pvc import tcnn_torch  # noqa: E402

PVC_RF_FILES = ("rift_pvc/radar_fields_upstream.py", "rift_pvc/radar_fields_training.py",
                "rift_pvc/radar_fields_gotcha.py", "train_radar_fields_pvc.py",
                *sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "rift_pvc/tcnn_torch").glob("*.py")))
CUDA_RF_FILES = ("train_radar_fields.py", "rift/radar_fields_upstream.py", "rift/radar_fields_gotcha.py",
                 "rift/radar_fields_recipe.py", "rift/radar_fields.py", "rift/radar_fields_native.py")
REFERENCE = ROOT / "external" / "RadarFields_reference"
needs_reference = pytest.mark.skipif(not REFERENCE.is_dir(), reason="pinned Radar Fields checkout not present")
DATA_ROOT = os.environ.get("RIFT_DATA_ROOT")
needs_data = pytest.mark.skipif(not DATA_ROOT or not Path(DATA_ROOT).is_dir(), reason="RIFT_DATA_ROOT not available")
TORCHSHIM = pvc_upstream.TORCHSHIM_BACKEND


@pytest.fixture(autouse=True)
def cpu_backend(monkeypatch):
    monkeypatch.setenv("RIFT_ACCELERATOR", "cpu")
    monkeypatch.setenv("RIFT_PVC_ALLOW_BACKEND", "cpu")
    monkeypatch.setenv("RIFT_PVC_TCNN_SHIM", "1")
    monkeypatch.delenv("RIFT_PVC_TCNN_HALF", raising=False)


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


def _production_command(tmp_path):
    cmd = commands_for("a320", "radar_fields", dataset_root=tmp_path, output_root=tmp_path, num_train=2400)[0]
    k = 1 if Path(cmd[0]).name.startswith("python") else 0
    return cmd, k


# --------------------------------------------------------------------------
# Adaptation surface
# --------------------------------------------------------------------------

def test_cuda_files_on_disk_are_untouched():
    for name in CUDA_RF_FILES:
        text = (ROOT / name).read_text()
        assert "rift_pvc" not in text and "xpu" not in text, name


def test_the_only_cuda_mentions_in_the_pvc_files_are_rng_state_and_device_count_under_cuda_guards():
    allowed = re.compile(r"torch\.cuda\.(device_count|get_rng_state_all|set_rng_state_all)\(")
    offenders = []
    for name in PVC_RF_FILES:
        for number, line in _code_lines(ROOT / name):
            if "torch.cuda." in line and not allowed.search(line):
                offenders.append(f"{name}:{number}: {line.strip()}")
            if re.search(r"autocast\(\s*[\"']cuda[\"']", line) or ".cuda()" in line or "torch.Generator(device" in line:
                offenders.append(f"{name}:{number}: {line.strip()}")
    assert offenders == []


def test_gotcha_backend_declaration_is_identical_to_the_cuda_entry_point():
    def literal(path):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "GOTCHA_BACKEND" for t in node.targets):
                return ast.literal_eval(node.value)
        raise AssertionError("GOTCHA_BACKEND not found")
    assert literal(ROOT / "train_radar_fields_pvc.py") == literal(ROOT / "train_radar_fields.py") == cuda_entry.GOTCHA_BACKEND
    assert pvc_entry.GOTCHA_BACKEND == cuda_entry.GOTCHA_BACKEND
    registry = gotcha_fe.backend_registry()["radar_fields"]
    assert registry["status"] == "available" and registry["module"] == "train_radar_fields_pvc"
    assert "from rift_pvc.radar_fields_gotcha import recipe_from_config" in (ROOT / "train_gotcha_dataset_pvc.py").read_text()


def test_install_rebinds_exactly_the_audited_names_and_is_idempotent():
    pvc_entry.install()
    pvc_entry.install()
    assert cuda_entry.set_seed is twins.set_seed
    assert cuda_entry.evaluate is twins.evaluate
    assert cuda_entry.checkpoint_payload is twins.checkpoint_payload
    assert cuda_entry.validate_resume_checkpoint is twins.validate_resume_checkpoint
    assert cuda_entry.build_model is twins.build_model
    assert cuda_entry.recipe_contract is twins.recipe_contract is radar_fields_recipe.recipe_contract
    assert cuda_entry.check_model_backend is pvc_upstream.check_model_backend is cuda_upstream.check_model_backend
    # the originals are still reachable for the twins' delegation
    assert twins._ORIGINAL["build_model"] is not twins.build_model
    # the copied main mirrors the original's control flow and reads the trainer's STOP flag
    import inspect
    pvc_main, cuda_main = inspect.getsource(pvc_entry.main), inspect.getsource(cuda_entry.main)
    for marker in ("Radar Fields reference commit", "Stopped cleanly after publishing checkpoint_latest.",
                   "diagnostic mode must not retain the full response payload", "coverage state disagrees",
                   "Nonfinite released RF loss gradient"):
        assert marker in cuda_main and marker in pvc_main
    assert "_rf.STOP_REQUESTED" in pvc_main and not re.search(r"(?<![._\w])STOP_REQUESTED", pvc_main)
    assert "twins.restore_device_rng_state(checkpoint, device, resume_validation)" in pvc_main


# --------------------------------------------------------------------------
# CLI and identity
# --------------------------------------------------------------------------

def test_parse_args_matches_the_original_except_device_default_and_backend_id(tmp_path):
    cmd, k = _production_command(tmp_path)
    assert cmd[cmd.index("--device") + 1] == "cuda"
    remapped = fe.remap_device(cmd)
    assert remapped[remapped.index("--device") + 1] == "cpu"     # the active backend under RIFT_ACCELERATOR=cpu
    theirs, mine = vars(cuda_entry.parse_args(cmd[k + 1:])), vars(pvc_entry.parse_args(remapped[k + 1:]))
    assert theirs.pop("device") == "cuda" and mine.pop("device") == "cpu"
    assert theirs.pop("model_backend") == "upstream-tcnn" and mine.pop("model_backend") == TORCHSHIM
    assert mine == theirs
    without_device = [t for i, t in enumerate(remapped[k + 1:]) if t != "--device" and remapped[k + 1:][i - 1] != "--device"]
    assert pvc_entry.parse_args(without_device).device == "cpu"
    # an explicit CUDA device keeps the real backend id: the twin never relabels a CUDA run
    assert pvc_entry.parse_args(cmd[k + 1:]).model_backend == "upstream-tcnn"
    with pytest.raises(SystemExit):
        pvc_entry.parse_args(remapped[k + 1:] + ["--model-backend", "upstream-tcnn"])
    assert pvc_entry.parse_args(remapped[k + 1:] + ["--model-backend", TORCHSHIM]).model_backend == TORCHSHIM
    with pytest.raises(ValueError, match="source-adapted-v3 fixes model_backend"):   # released network locked (RF2)
        pvc_entry.parse_args(remapped[k + 1:] + ["--model-backend=torch"])
    with pytest.raises(SystemExit):
        pvc_entry.parse_args(["--recipe", "legacy-v1", "--npz-path", "x.npz", "--model-backend", TORCHSHIM])


def test_recipe_contract_records_the_shim_identity_and_refuses_cross_backend_resume(tmp_path):
    cmd, k = _production_command(tmp_path)
    args = pvc_entry.parse_args(fe.remap_device(cmd)[k + 1:])
    contract = radar_fields_recipe.recipe_contract(args)
    assert contract["model_backend"] == TORCHSHIM and contract["encoding"] == "original_radarfield_tcnn_torchshim"
    assert contract["tcnn_shim"] == {"version": tcnn_torch.SHIM_VERSION, "precision": "fp32"}
    mirror = copy.copy(args)
    mirror.model_backend = "upstream-tcnn"
    original = twins._ORIGINAL["recipe_contract"](mirror)
    assert {k: v for k, v in contract.items() if k not in ("model_backend", "encoding", "tcnn_shim")} == \
           {k: v for k, v in original.items() if k not in ("model_backend", "encoding")}
    assert original["encoding"] == "original_radarfield_tcnn"
    cuda_checkpoint = {"args": vars(mirror), "radar_fields_recipe": original}
    with pytest.raises(ValueError, match="mismatch"):
        radar_fields_recipe.validate_recipe_checkpoint(cuda_checkpoint, args)
    radar_fields_recipe.validate_recipe_checkpoint({"args": vars(args), "radar_fields_recipe": contract}, args)
    with pytest.raises(ValueError, match="upstream-tcnn or torch"):
        twins._ORIGINAL["recipe_contract"](args)   # the unchanged CUDA contract refuses the PVC identity


def test_check_model_backend_twin_gates_devices_and_installs_the_shim(monkeypatch):
    base = dict(sh_degree=3, hash_features=2, hash_base_resolution=16, hash_log2_size=19)
    args = SimpleNamespace(model_backend=TORCHSHIM, device="cpu", **base)
    monkeypatch.delenv("RIFT_PVC_TCNN_SHIM")
    with pytest.raises(ValueError, match="RIFT_PVC_TCNN_SHIM"):
        pvc_upstream.check_model_backend(args)
    monkeypatch.setenv("RIFT_PVC_TCNN_SHIM", "1")
    pvc_upstream.check_model_backend(args)
    assert sys.modules["tinycudann"] is tcnn_torch and pvc_upstream.shim_active()
    with pytest.raises(ValueError, match="fixes hash_log2_size=19"):
        pvc_upstream.check_model_backend(SimpleNamespace(model_backend=TORCHSHIM, device="cpu", **dict(base, hash_log2_size=18)))
    with pytest.raises(ValueError, match="does not run on cuda"):
        pvc_upstream.check_model_backend(SimpleNamespace(model_backend=TORCHSHIM, device="cuda", **base))
    with pytest.raises(RuntimeError, match="XPU"):
        pvc_upstream.check_model_backend(SimpleNamespace(model_backend=TORCHSHIM, device="xpu", **base))
    with pytest.raises(ValueError, match="requires CUDA"):
        pvc_upstream.check_model_backend(SimpleNamespace(model_backend="upstream-tcnn", device="cpu", **base))
    pvc_upstream.check_model_backend(SimpleNamespace(model_backend="torch", device="cpu", **base))


# --------------------------------------------------------------------------
# The release through the shim, checkpoint identity
# --------------------------------------------------------------------------

@needs_reference
def test_original_radarfield_runs_through_the_shim_on_cpu_and_checkpoints_carry_the_identity(tmp_path):
    args = pvc_entry.parse_args(["--recipe", "source-adapted-v3", "--npz-path", "unused.npz", "--device", "cpu",
                                 "--granularity", "6"])
    assert args.model_backend == TORCHSHIM
    model = twins.build_model(args, torch.device("cpu"))
    assert isinstance(model, cuda_upstream.OriginalRadarFieldsModel)
    assert isinstance(model.original.encode_xyz, tcnn_torch.HashGridEncoding)
    assert isinstance(model.original.encode_angle, tcnn_torch.SphericalHarmonicsEncoding)
    assert isinstance(model.original.xyz_net[0], tcnn_torch.FullyFusedMLP)
    assert isinstance(model.original.xyz_net[1], torch.nn.BatchNorm1d)
    groups = model.get_params(args.lr)
    assert [sum(p.numel() for p in g["params"]) for g in groups] == [10523376, 0, 4096 + 32 + 32, 3072, 4096]   # BN1d(32) weight+bias
    assert [k for k in model.state_dict() if k.endswith(".params")] == [
        "original.encode_xyz.params", "original.encode_angle.params", "original.xyz_net.0.params",
        "original.alpha_net.params", "original.rd_net.params"]
    model.eval()
    xyz = (torch.rand(257, 3) - .5) * args.extent
    direction = torch.nn.functional.normalize(torch.randn_like(xyz), dim=-1)
    direct = model.original((xyz + args.extent) / (2 * args.extent), direction, sin_epoch=.8)
    wrapped = model(xyz, direction, mask_progress=.8)
    torch.testing.assert_close(wrapped["alpha"], direct["alpha"].flatten(), rtol=0, atol=0)
    torch.testing.assert_close(wrapped["reflectance"], direct["rd"].flatten(), rtol=0, atol=0)
    assert wrapped["alpha"].dtype == torch.float32 and (wrapped["alpha"] > 0).all() and (wrapped["alpha"] < 1).all()
    wrapped["rcs"].square().mean().backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients) and any(g.abs().max() > 0 for g in gradients)
    chunked = model.query_chunked(xyz, direction, mask_progress=.8, chunk_size=53)
    for key in wrapped:
        torch.testing.assert_close(chunked[key], wrapped[key], rtol=5e-3, atol=5e-4)
    # checkpoint payload twin: original schema plus the PVC identity
    optimizer = torch.optim.Adam(groups, lr=args.lr, betas=(0.9, 0.99), eps=1e-15)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    grid = (torch.rand(args.granularity ** 3, 3) - .5) * args.extent
    stats = {"peak_power": 1.0, "dynamic_range_db": 60.0}
    payload = twins.checkpoint_payload(model, optimizer, scheduler, 3, 0.5, grid, stats, np.random.default_rng(0), [], args)
    assert payload["accelerator_backend"] == "cpu" and payload["cuda_rng_state"] is None and "xpu_rng_state" not in payload
    assert payload["tcnn_shim"] == tcnn_torch.identity() and payload["args"]["model_backend"] == TORCHSHIM
    assert payload["radar_fields_recipe"]["model_backend"] == TORCHSHIM
    assert set(payload) >= {"radar_fields_state_dict", "optimizer_state_dict", "torch_rng_state", "numpy_rng_state_json",
                            "resume_contract_version", "training_view_coverage"} - {"training_view_coverage"}
    # the same checkpoint reloads into a fresh shim model bit for bit
    fresh = twins.build_model(args, torch.device("cpu"))
    fresh.load_state_dict(payload["radar_fields_state_dict"])
    for (name, a), (_, b) in zip(model.state_dict().items(), fresh.state_dict().items()):
        assert torch.equal(a, b), name


def _restore_probe(monkeypatch):
    calls = {}
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda s: calls.setdefault("cuda", s))
    return calls


def test_restore_device_rng_state_uses_the_payload_of_the_active_backend(monkeypatch):
    strict = {"strict_resume_contract": True}
    calls = _restore_probe(monkeypatch)
    state = [torch.zeros(16, dtype=torch.uint8)]
    twins.restore_device_rng_state({"cuda_rng_state": state}, torch.device("cpu"), strict)
    assert calls == {}
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    twins.restore_device_rng_state({"cuda_rng_state": state}, torch.device("cuda"), strict)
    assert "cuda" in calls
    with pytest.raises(ValueError, match="XPU"):
        twins.normalize_device_rng_state(None, require_present=True, backend="xpu")
    with pytest.raises(ValueError, match="topology"):
        twins.normalize_device_rng_state(state * 2, expected_device_count=1, backend="xpu")
    assert twins.normalize_device_rng_state(None, require_present=False, backend="xpu") is None


# --------------------------------------------------------------------------
# Native GOTCHA twin
# --------------------------------------------------------------------------

def _gotcha_config():
    # torchshim fixes hash_log2_size/base/features/sh_degree like upstream-tcnn; widths must be FullyFusedMLP widths.
    return dict(profile="audited-v2", model_backend=TORCHSHIM, steps=2, view_batch=1, seed=7, ray_samples=8,
                eval_every=2, checkpoint_every=1, hidden_dim=16, feature_dim=4, hash_levels=2, hash_final_resolution=8)


def test_gotcha_twin_defaults_and_recipe_identity():
    assert pvc_gotcha.DEFAULTS["model_backend"] == TORCHSHIM
    assert {k: v for k, v in pvc_gotcha.DEFAULTS.items() if k != "model_backend"} == \
           {k: v for k, v in cuda_gotcha.DEFAULTS.items() if k != "model_backend"}
    recipe = pvc_gotcha.recipe_from_config({}, .15, 2400)
    assert recipe["controls"]["model_backend"] == TORCHSHIM and recipe["model_recipe"]["model_backend"] == TORCHSHIM
    assert recipe["model_recipe"]["tcnn_shim"]["version"] == tcnn_torch.SHIM_VERSION
    assert (recipe["controls"]["steps"], recipe["controls"]["eval_every"]) == (960, 240)
    for key in ("target", "range_grid", "occupancy", "normalization", "exposure", "fidelity"):
        assert recipe[key] == cuda_gotcha.recipe_from_config({}, .15, 2400)[key]


@needs_reference
def test_gotcha_twin_trains_interrupts_and_resumes_on_cpu(tmp_path, monkeypatch):
    from test_gotcha_dataset import write_shard, tiny_region
    from rift.gotcha_dataset import GOTCHADataset
    write_shard(tmp_path / "New_Transfer/shards/pass1_hh.npz", 1, "hh", nf=33)
    ds = GOTCHADataset(tmp_path, passes=(1,), polarizations=("hh",), region=tiny_region())
    config = _gotcha_config()
    complete = tmp_path / "complete"
    result = pvc_gotcha.run_gotcha(dataset=ds, output_dir=complete, config=config, device="cpu", resume=None)
    assert result["status"] == "complete" and result["step"] == 2 and result["test_accessed"] is False
    full = torch.load(complete / "checkpoint_final.pt", weights_only=False)
    assert full["rng_cuda"] == [] and full["rng_xpu"] == [] and full["accelerator_backend"] == "cpu"
    assert full["tcnn_shim"] == tcnn_torch.identity() and full["recipe"]["controls"]["model_backend"] == TORCHSHIM
    assert "hh.original.encode_xyz.params" in full["model_state_dict"]
    interrupted = tmp_path / "interrupted"
    original_step = torch.optim.Adam.step
    with monkeypatch.context() as m:
        def stop_after_update(opt, *args, **kwargs):
            out = original_step(opt, *args, **kwargs)
            signal.raise_signal(signal.SIGTERM)
            return out
        m.setattr(torch.optim.Adam, "step", stop_after_update)
        partial = pvc_gotcha.run_gotcha(dataset=ds, output_dir=interrupted, config=config, device="cpu", resume=None)
        assert partial["status"] == "interrupted" and partial["step"] == 1
    result = pvc_gotcha.run_gotcha(dataset=ds, output_dir=interrupted, config=config, device="cpu",
                                   resume=interrupted / "checkpoint_latest.pt")
    assert result["status"] == "complete"
    resumed = torch.load(interrupted / "checkpoint_final.pt", weights_only=False)
    for key in ("history", "training_view_coverage", "rng_numpy"):
        assert full[key] == resumed[key], key
    for name, value in full["model_state_dict"].items():
        assert torch.equal(value, resumed["model_state_dict"][name]), name
    assert torch.equal(full["rng_torch"], resumed["rng_torch"])
    bad = dict(full); bad.pop("rng_xpu")
    bad_path = tmp_path / "no_xpu.pt"; torch.save(bad, bad_path)
    with pytest.raises(ValueError, match="rng_xpu"):
        pvc_gotcha.run_gotcha(dataset=ds, output_dir=tmp_path / "bad", config=config, device="cpu", resume=bad_path)
    with pytest.raises(ValueError, match="recipe|Unknown"):
        pvc_gotcha.run_gotcha(dataset=ds, output_dir=tmp_path / "wrong", config={**config, "ray_samples": 9},
                              device="cpu", resume=complete / "checkpoint_final.pt")


# --------------------------------------------------------------------------
# Frontends
# --------------------------------------------------------------------------

def test_frontend_device_remap_and_smoke_marking():
    assert fe.remap_device(["x.py", "--device", "cuda:0", "--steps", "960"]) == ["x.py", "--device", "cpu", "--steps", "960"]
    assert fe.remap_device(["x.py", "--device", "xpu"]) == ["x.py", "--device", "xpu"]
    assert fe.remap_device(["x.py", "--steps", "1"]) == ["x.py", "--steps", "1"]
    assert fe.mark_smoke(["x.py", "--checkpoint-name", "radar_fields"]) == ["x.py", "--checkpoint-name", "radar_fields_pvcsmoke"]
    with pytest.raises(ValueError):
        fe.mark_smoke(["x.py", "--steps", "1"])
    with pytest.raises(SystemExit):
        fe.parse_args(["--pvc-epochs", "2", "--pvc-smoke", "--list"])


@needs_data
def test_dry_run_plan_for_radar_fields_rewrites_the_device_and_marks_the_smoke(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(ROOT), RIFT_ACCELERATOR="cpu")
    proc = subprocess.run([sys.executable, "train_rift_dataset_pvc.py", "--dataset-root", DATA_ROOT, "--output-root",
                           str(tmp_path / "out"), "--object", "b787", "--method", "radar_fields", "--num-train", "2400",
                           "--num-tx", "1", "--num-rx", "1", "--dry-run", "--pvc-smoke"],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-1500:]
    plan = json.loads(proc.stdout[proc.stdout.index("{"):])
    entry = plan["plans"][0]
    cmd, cuda = entry["commands"][0], entry["cuda_commands"][0]
    k = 1 if Path(cmd[0]).name.startswith("python") else 0
    assert Path(cmd[k]).name == "train_radar_fields_pvc.py" and Path(cuda[k]).name == "train_radar_fields.py"
    assert cuda[cuda.index("--device") + 1] == "cuda" and cmd[cmd.index("--device") + 1] == "cpu"
    assert cmd[cmd.index("--checkpoint-name") + 1] == "radar_fields_pvcsmoke" and cmd[cmd.index("--steps") + 1] == "960"
    assert entry["output_dir"].endswith("_pvcsmoke") and plan["pvc_smoke"] is True
    strip = lambda c: [t for i, t in enumerate(c) if i == 0 or c[i - 1] not in ("--device", "--checkpoint-name")]
    assert strip(cmd)[k + 1:] == strip(cuda)[k + 1:]
