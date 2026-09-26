"""RadarSplat PVC backend, engine twins, entry point and frontend wiring (Package F). CPU."""
from __future__ import annotations

import ast
import copy
import json
import os
import signal
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rift.radarsplat_release import REFERENCE_ROOT  # noqa: E402
from rift_pvc import radarsplat_xpu_backend as backend  # noqa: E402

pytestmark = pytest.mark.skipif(not (REFERENCE_ROOT / "gsplat/rendering.py").is_file(),
                                reason="pinned RadarSplat source not staged (scripts/fetch_radarsplat_reference.py)")
DATA_ROOT = os.environ.get("RIFT_DATA_ROOT")
needs_data = pytest.mark.skipif(not DATA_ROOT or not Path(DATA_ROOT).is_dir(), reason="RIFT_DATA_ROOT not available")
ENV = dict(os.environ, PYTHONPATH=str(ROOT), RIFT_PVC_ALLOW_BACKEND="cpu", RIFT_ACCELERATOR="cpu")


@pytest.fixture(scope="module")
def reference():
    return backend.load_xpu_reference(device="cpu")


def small_grid():
    from rift.radarsplat_b7873200 import RadarSplatGrid
    return RadarSplatGrid(num_range_bins=16, range_resolution_m=0.1, range_start_m=9.2, azimuth_start_deg=-7.2,
                          azimuth_span_deg=14.4, output_azimuth_resolution_deg=0.9,
                          intermediate_azimuth_resolution_deg=0.1, spectral_leakage_width_m=0.7)


def facing_pose():
    pose = torch.eye(4)
    pose[:3, 3] = torch.tensor([-10.0, 0.0, 0.0])  # sensor 10 m out, x-axis toward the scene
    return pose


def test_load_xpu_reference_binds_mirrors_and_guards_every_cuda_op(reference):
    rendering, fused_ssim = reference
    from rift_pvc import gsplat_torch_ops as ops
    from rift_pvc.fused_ssim_torch import fused_ssim as twin
    import gsplat.cuda._wrapper as wrapper
    assert fused_ssim is twin
    for name, mirror in backend.RENDERING_REBINDS.items():
        assert getattr(rendering, name) is mirror and getattr(wrapper, name) is mirror
    assert wrapper.rasterize_to_indices_in_range_radargs is ops.rasterize_to_indices_in_range_radargs_xpu
    for name in backend.RENDERING_DEVICE_LITERALS:
        assert getattr(getattr(rendering, name), backend.DEVICE_NEUTRAL_MARK)
    stub = sys.modules[backend.BACKEND_MODULE]
    with pytest.raises(RuntimeError, match="rasterize_to_pixels_fwd.*reached on the PVC backend"):
        stub._C.rasterize_to_pixels_fwd
    with pytest.raises(RuntimeError, match="reached on the PVC backend"):
        wrapper.rasterize_to_pixels(torch.zeros(1, 1, 2), torch.zeros(1, 1, 3), torch.zeros(1, 1, 3), torch.ones(1, 1),
                                    16, 16, 16, torch.zeros(1, 1, 1, dtype=torch.int32),
                                    torch.zeros(0, dtype=torch.int32))
    with pytest.raises(ValueError, match="PVC backend"):
        backend.load_xpu_reference(device="cuda")
    assert Path(rendering.__file__).resolve().is_relative_to(REFERENCE_ROOT)


def test_device_neutral_reexecution_asserts_the_literal_counts(tmp_path):
    source = tmp_path / "fake_fork.py"
    source.write_text("import torch\n\ndef f(x):\n    return torch.ones(2).to('cuda') + x\n")
    module = types.ModuleType("fake_fork")
    module.__file__ = str(source)
    exec(compile(source.read_text(), str(source), "exec"), module.__dict__)
    with pytest.raises(RuntimeError, match="expected exactly 2 literal"):
        backend.reexecute_device_neutral(module, "f", {".to('cuda')": (".to(x.device)", 2)})
    rebuilt = backend.reexecute_device_neutral(module, "f", {".to('cuda')": (".to(x.device)", 1)})
    assert rebuilt is module.f and torch.equal(module.f(torch.zeros(2)), torch.ones(2))
    assert backend.reexecute_device_neutral(module, "f", {}) is rebuilt  # idempotent


def test_preprocessing_fft_runs_on_the_requested_device(reference):
    functions = backend.preprocessing_functions("cpu")
    image = np.random.default_rng(0).random((12, 33))
    log_image, magnitude, fft, frequencies = functions["FFT"](image, range_resolution=0.05)
    np.testing.assert_allclose(magnitude, np.abs(np.fft.fft(image, axis=1)), rtol=1e-5, atol=1e-6)
    assert callable(functions["multipath_modeling"])


def test_released_renderer_end_to_end_on_the_mirrors(reference):
    rendering, fused_ssim = reference
    from rift_pvc.radarsplat_release import ReleasedRenderer, create_scene, release_loss
    splats, _ = create_scene(scene_scale=100, scene_center=[0, 0, 0], device="cpu", num_points=500)
    renderer = ReleasedRenderer(rendering, 100.0)
    power, occupancy = renderer(splats, facing_pose(), small_grid(), 5, torch.zeros(16, 16))
    assert power.shape == occupancy.shape == (16, 16)
    assert torch.isfinite(power).all() and float(power.max()) > 1e-3 and float(occupancy.max()) > 1e-3
    target = torch.rand(16, 16)
    losses = release_loss(power, occupancy, target, (target > .5).float(), splats, fused_ssim)
    losses["total"].backward()
    for name, parameter in splats.items():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert float(splats["means"].grad.abs().max()) > 0 and float(splats["sh0"].grad.abs().max()) > 0


@pytest.fixture
def synthetic_cache(tmp_path):
    from scripts.validate_radarsplat_b7873200_native_contract import _make_synthetic_cache
    from rift.radarsplat_b7873200_protocol import load_cache
    _make_synthetic_cache(tmp_path / "cache")
    return load_cache(tmp_path / "cache")


@pytest.fixture
def tiny_engine(monkeypatch, reference):
    """The PVC engine twin with a 64-Gaussian scene and a SIGTERM raised on the n-th render."""
    import rift.radarsplat_release_training as engine
    import rift_pvc.radarsplat_release_training as pvc_engine
    import rift.radarsplat_release as release
    original_create = release.create_scene
    monkeypatch.setattr(engine, "create_scene", lambda **kw: original_create(**{**kw, "num_points": 64}))
    control = dict(calls=0, stop=1)
    Real = engine.ReleasedRenderer

    class Stopper(Real):
        def __call__(self, *args, **kwargs):
            control["calls"] += 1
            if control["calls"] == control["stop"]:
                signal.raise_signal(signal.SIGTERM)
            return super().__call__(*args, **kwargs)

    monkeypatch.setattr(engine, "ReleasedRenderer", Stopper)
    monkeypatch.setenv("RIFT_PVC_RADARSPLAT_TIMING_EVERY", "1")
    pvc_engine.install()
    return pvc_engine, control


def run_leg(pvc_engine, control, cache, folder, stop, *, resume=True):
    import train_radarsplat as lifecycle
    control.update(calls=0, stop=stop)
    with pytest.raises(SystemExit) as exc:
        pvc_engine.train(cache, folder, device=torch.device("cpu"), resume=resume, profile="budget48")
    assert exc.value.code == 143
    return lifecycle._load_checkpoint(folder / "checkpoint_latest.pt", torch.device("cpu"))


def test_engine_interrupt_resume_sidecar_and_cuda_checkpoint_refusal(tmp_path, synthetic_cache, tiny_engine, capsys):
    import train_radarsplat as lifecycle
    pvc_engine, control = tiny_engine
    cache = synthetic_cache
    first = run_leg(pvc_engine, control, cache, tmp_path / "resume", 1, resume=False)
    sidecar = backend.read_sidecar(tmp_path / "resume")
    assert sidecar["radarsplat_backend"] == backend.BACKEND_IDENTITY and sidecar["ssim"] == backend.SSIM_IDENTITY
    assert sidecar["schema"] == backend.SIDECAR_SCHEMA and "F-dev1" in " ".join(sidecar["deviations"])
    assert "radarsplat_backend" not in json.dumps(first["identity"])  # D5: identity dict untouched
    resumed = run_leg(pvc_engine, control, cache, tmp_path / "resume", 1)
    full = run_leg(pvc_engine, control, cache, tmp_path / "full", 2, resume=False)
    assert first["step"] == 1 and resumed["step"] == full["step"] == 2
    for key in ("splats", "optimizers", "position_scheduler", "sampler"):
        assert lifecycle._directly_equal(resumed[key], full[key]), key
    assert "RADARSPLAT_PVC_UPDATE_TIMING_JSON=" in capsys.readouterr().out
    # a checkpoint directory without the sidecar is a CUDA-renderer run: refused unless overridden
    (tmp_path / "resume" / backend.SIDECAR_NAME).unlink()
    with pytest.raises(RuntimeError, match="released CUDA renderer"):
        pvc_engine.train(cache, tmp_path / "resume", device=torch.device("cpu"), resume=True, profile="budget48")
    foreign = dict(backend.backend_record(), radarsplat_backend="fork_sycl_port_v1")
    (tmp_path / "resume" / backend.SIDECAR_NAME).write_text(json.dumps(foreign))
    with pytest.raises(RuntimeError, match="fork_sycl_port_v1"):
        pvc_engine.train(cache, tmp_path / "resume", device=torch.device("cpu"), resume=True, profile="budget48")
    with pytest.raises(RuntimeError, match="names backend"):
        backend.write_sidecar(tmp_path / "resume")
    # readout on the PVC backend carries both records
    (tmp_path / "resume" / backend.SIDECAR_NAME).write_text(json.dumps(backend.backend_record()))
    torch.save(resumed, tmp_path / "resume" / "checkpoint_latest.pt")
    control.update(calls=0, stop=-1)
    result = pvc_engine.readout(resumed, checkpoint_path=tmp_path / "resume" / "checkpoint_latest.pt",
                                cache_root=cache.root, device=torch.device("cpu"), role="validation",
                                geometry_path=tmp_path / "geometry.npz")
    assert result["step"] == 2 and result["pvc"]["readout_backend"] == backend.BACKEND_IDENTITY
    assert result["pvc"]["checkpoint_backend"]["radarsplat_backend"] == backend.BACKEND_IDENTITY
    with np.load(tmp_path / "geometry.npz") as data:
        assert data["support"].shape == (48, 48, 48)


def test_engine_install_is_idempotent_and_originals_are_kept():
    import rift.radarsplat_release_training as engine
    import train_radarsplat as lifecycle
    import rift_pvc.radarsplat_release_training as pvc_engine
    pvc_engine.install()
    pvc_engine.install()
    assert engine.train is pvc_engine.train and engine.load_cuda_reference is backend.load_xpu_reference
    assert engine.ReleasedPreprocessing.__module__ == "rift_pvc.radarsplat_release"
    assert lifecycle._atomic_torch_save is pvc_engine._atomic_torch_save
    assert pvc_engine.ORIGINAL["train"] is not pvc_engine.train
    assert pvc_engine.ORIGINAL["train"].__module__ == "rift.radarsplat_release_training"


def _literal(path, name):
    tree = ast.parse(Path(path).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(name)


def test_entry_point_declaration_matches_the_original_required_keys():
    pvc = _literal(ROOT / "train_radarsplat_pvc.py", "GOTCHA_BACKEND")
    cuda = _literal(ROOT / "train_radarsplat.py", "GOTCHA_BACKEND")
    for key in ("schema", "method", "callable", "selection_unit", "joint_passes", "native_frequency_policy",
                "polarizations", "metric_domain"):
        assert pvc[key] == cuda[key], key
    assert "fork_torch_mirror_xpu_v1" in pvc["runtime_requirements"] and "pvc" in pvc["fidelity_status"]
    import train_radarsplat_pvc as entry
    assert entry.GOTCHA_BACKEND == pvc and callable(entry.run_gotcha)


@pytest.mark.parametrize("profile", ["budget48", "legacy"])
def test_entry_point_help_runs_the_real_parsers(profile):
    proc = subprocess.run([sys.executable, "train_radarsplat_pvc.py", "--fidelity-profile", profile, "--help"],
                          cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-1500:]
    assert "--checkpoint-dir" in proc.stdout and "--device" in proc.stdout
    if profile == "budget48":
        assert "--steps" not in proc.stdout  # the release CLI has no step flag
    else:
        assert "--steps" in proc.stdout


def test_entry_point_refuses_without_an_xpu_unless_overridden():
    env = dict(os.environ, PYTHONPATH=str(ROOT), RIFT_ACCELERATOR="cpu")
    env.pop("RIFT_PVC_ALLOW_BACKEND", None)
    proc = subprocess.run([sys.executable, "train_radarsplat_pvc.py", "--fidelity-profile", "budget48", "--help"],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode != 0 and "RIFT_PVC_ALLOW_BACKEND" in proc.stderr


def test_legacy_parser_injects_the_accelerator_device(monkeypatch):
    monkeypatch.setenv("RIFT_ACCELERATOR", "cpu")
    import train_radarsplat_pvc as entry
    args = entry.parse_args(["--cache-root", "c", "--checkpoint-dir", "d"])
    assert args.device == "cpu"
    args = entry.parse_args(["--cache-root", "c", "--checkpoint-dir", "d", "--device", "xpu:1"])
    assert args.device == "xpu:1"
    released = entry.parse_args(["--fidelity-profile", "budget48", "--cache-root", "c", "--checkpoint-dir", "d"])
    assert released.device == "cpu" and released.steps == 2000 and released.init_num_gaussians == 112000


def test_prepare_twin_injects_device_and_reports_accelerator_memory(monkeypatch, capsys):
    monkeypatch.setenv("RIFT_ACCELERATOR", "cpu")
    import scripts_pvc.prepare_radarsplat_b7873200_targets_pvc as twin
    twin.install()
    args = twin.parse_args(["--cache-root", "x", "--npz-path", "a.npz", "--role-manifest", "m.json"])
    assert args.device == "cpu"
    twin._emit_prepare_resource(device=torch.device("cpu"), materialized=3, reused=1)
    line = [l for l in capsys.readouterr().out.splitlines() if l.startswith("RADARSPLAT_B7873200_PREPARE_RESOURCE_JSON=")][0]
    payload = json.loads(line.split("=", 1)[1])
    assert payload["targets_newly_materialized"] == 3 and "cuda_max_memory_allocated_bytes" not in payload
    assert twin._prepare._emit_prepare_resource is twin._emit_prepare_resource


def test_frontend_maps_both_radarsplat_commands():
    import train_rift_dataset_pvc as fe
    assert fe.to_pvc_command(["train_radarsplat.py", "--x"]) == ["train_radarsplat_pvc.py", "--x"]
    assert fe.to_pvc_command(["scripts/prepare_radarsplat_b7873200_targets.py", "--y"]) == [
        "scripts_pvc/prepare_radarsplat_b7873200_targets_pvc.py", "--y"]


def test_gotcha_pvc_registry_lists_radarsplat_as_available():
    proc = subprocess.run([sys.executable, "train_gotcha_dataset_pvc.py", "--list"], cwd=ROOT, env=ENV,
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-1500:]
    registry = json.loads(proc.stdout[proc.stdout.index("{"):])
    entry = registry["radarsplat"]
    assert entry["status"] == "available" and entry["module"] == "train_radarsplat_pvc"
    assert entry["polarizations"] == ["hh", "hv", "vh", "vv"]


@needs_data
def test_dry_run_plan_matches_cuda_plan_except_entrypoints(tmp_path):
    common = ["--dataset-root", DATA_ROOT, "--output-root", str(tmp_path / "out"), "--object", "b787",
              "--method", "radarsplat", "--radarsplat-recipe", "budget48", "--num-train", "2400",
              "--num-tx", "1", "--num-rx", "1", "--dry-run"]
    cuda = subprocess.run([sys.executable, "train_rift_dataset.py", *common], cwd=ROOT, env=ENV,
                          capture_output=True, text=True, timeout=900)
    pvc = subprocess.run([sys.executable, "train_rift_dataset_pvc.py", *common], cwd=ROOT, env=ENV,
                         capture_output=True, text=True, timeout=900)
    assert cuda.returncode == 0, cuda.stderr[-1500:]
    assert pvc.returncode == 0, pvc.stderr[-1500:]
    c = json.loads(cuda.stdout[cuda.stdout.index("{"):])["plans"][0]
    p = json.loads(pvc.stdout[pvc.stdout.index("{"):])["plans"][0]
    assert p["cuda_commands"] == c["commands"] and len(p["commands"]) == 2
    names = [Path(cmd[1]).name for cmd in p["commands"]]
    assert names == ["prepare_radarsplat_b7873200_targets_pvc.py", "train_radarsplat_pvc.py"]
    for ccmd, pcmd in zip(c["commands"], p["commands"]):
        assert ccmd[0] == pcmd[0] and ccmd[2:] == pcmd[2:] and "--device" not in pcmd
    assert p["commands"][1][-3:] == ["--fidelity-profile", "budget48", "--no-resume"]


def test_gotcha_wrapper_refuses_completed_heads_without_the_sidecar(tmp_path, monkeypatch):
    """Audit F3: a completed head without backend.json (a CUDA-renderer head) is refused before any head runs."""
    import rift_pvc.radarsplat_gotcha as gotcha
    head = tmp_path / "run" / "hh" / "checkpoints"
    head.mkdir(parents=True)
    (head / "checkpoint_final.pt").write_bytes(b"x")
    (head / "checkpoint_latest.pt").write_bytes(b"x")
    dataset = types.SimpleNamespace(polarizations=("hh",))
    monkeypatch.setattr(gotcha._backend, "run_gotcha", lambda **_: pytest.fail("the original ran before the head gate"))
    with pytest.raises(RuntimeError, match="released CUDA renderer"):
        gotcha.run_gotcha(dataset=dataset, output_dir=tmp_path / "run", config={}, device="cpu", resume=None)
    backend.write_sidecar(head)
    monkeypatch.setattr(gotcha._backend, "run_gotcha", lambda **_: dict(status="complete"))
    result = gotcha.run_gotcha(dataset=dataset, output_dir=tmp_path / "run", config={}, device="cpu", resume=None)
    assert result["pvc"]["radarsplat_backend"] == backend.BACKEND_IDENTITY
    # the run root carries the sidecar after the call (conversion-only states included)
    assert backend.read_sidecar(tmp_path / "run")["radarsplat_backend"] == backend.BACKEND_IDENTITY
    foreign = dict(backend.backend_record(), radarsplat_backend="fork_sycl_port_v1")
    (head / backend.SIDECAR_NAME).write_text(json.dumps(foreign))
    with pytest.raises(RuntimeError, match="fork_sycl_port_v1"):
        gotcha.run_gotcha(dataset=dataset, output_dir=tmp_path / "run", config={}, device="cpu", resume=None)


def test_collection_readout_cli_runs_the_pvc_engine(tmp_path, synthetic_cache, tiny_engine):
    """Audit F4: the collection readout twin reads a released checkpoint on the PVC backend."""
    import scripts_pvc.readout_radarsplat_checkpoint_pvc as cli
    pvc_engine, control = tiny_engine
    run_leg(pvc_engine, control, synthetic_cache, tmp_path / "run", 1, resume=False)
    control.update(calls=0, stop=-1)
    output = tmp_path / "readout.json"
    cli.main(["--checkpoint", str(tmp_path / "run" / "checkpoint_latest.pt"), "--cache-root", str(synthetic_cache.root),
              "--device", "cpu", "--output", str(output)])
    result = json.loads(output.read_text())
    assert result["step"] == 1 and result["schema"] == "rift_radarsplat_released_readout_v1"
    assert result["pvc"]["readout_backend"] == backend.BACKEND_IDENTITY
    assert result["pvc"]["checkpoint_backend"]["radarsplat_backend"] == backend.BACKEND_IDENTITY
    assert result["pvc"]["entrypoint"].endswith("readout_radarsplat_checkpoint_pvc.py")
