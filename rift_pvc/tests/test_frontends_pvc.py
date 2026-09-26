"""PVC dataset frontends: same planning as the CUDA frontends, `_pvc` entry points."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_rift_dataset_pvc as fe  # noqa: E402

DATA_ROOT = os.environ.get("RIFT_DATA_ROOT")
needs_data = pytest.mark.skipif(not DATA_ROOT or not Path(DATA_ROOT).is_dir(), reason="RIFT_DATA_ROOT not available")


def test_to_pvc_command_maps_and_refuses():
    assert fe.to_pvc_command(["train.py", "--x", "1"]) == ["train_pvc.py", "--x", "1"]
    with pytest.raises(ValueError):
        fe.to_pvc_command(["train_unknown.py"])
    missing = [k for k, v in fe.PVC_ENTRYPOINTS.items() if not (ROOT / v).is_file()]
    if missing:
        with pytest.raises(FileNotFoundError):
            fe.to_pvc_command([missing[0]])
    absolute = [sys.executable, str(ROOT / "train.py"), "--y"]
    assert fe.to_pvc_command(absolute) == [sys.executable, str(ROOT / "train_pvc.py"), "--y"]


def test_bound_for_smoke_marks_names():
    cmd = ["train_pvc.py", "--epochs", "150", "--checkpoint-name", "rift", "--execution-contract-label", "lbl"]
    out = fe.bound_for_smoke(cmd, 2)
    assert out[out.index("--epochs") + 1] == "2"
    assert out[out.index("--checkpoint-name") + 1] == "rift" + fe.SMOKE_SUFFIX
    assert out[out.index("--execution-contract-label") + 1] == "lbl" + fe.SMOKE_SUFFIX
    assert cmd[cmd.index("--epochs") + 1] == "150"  # input untouched
    with pytest.raises(ValueError):
        fe.bound_for_smoke(["train_pvc.py", "--lr", "1"], 2)
    with pytest.raises(ValueError):
        fe.bound_for_smoke(["train_pvc.py", "--epochs", "150"], 2)


@pytest.mark.parametrize("script,method", [("train_spinr_style.py", "spinr"),
                                          ("train_sugavanam_ertin.py", "sugavanam_ertin"),
                                          ("train_geraf.py", "geraf"),
                                          ("train_sh_sas.py", "sh_sas")])
def test_smoke_epoch_budget_rejects_unsupported_methods(monkeypatch, script, method):
    from types import SimpleNamespace
    plan = {"plans": [{"method": method, "output_dir": "/tmp/audit/" + method,
                        "commands": [[sys.executable, str(ROOT / script)]]}]}
    monkeypatch.setattr(fe.base, "make_plan", lambda args: plan)
    with pytest.raises(ValueError, match="supports only RIFT/grid/isotropic"):
        fe.make_plan(SimpleNamespace(pvc_epochs=2, pvc_smoke=False))
    assert plan["plans"][0]["output_dir"] == "/tmp/audit/" + method


@needs_data
def test_dry_run_plan_matches_cuda_plan_except_entrypoint(tmp_path):
    common = ["--dataset-root", DATA_ROOT, "--output-root", str(tmp_path / "out"), "--object", "b787",
              "--method", "rift", "--num-train", "2400", "--num-tx", "1", "--num-rx", "1", "--dry-run"]
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    cuda = subprocess.run([sys.executable, "train_rift_dataset.py", *common], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=600)
    pvc = subprocess.run([sys.executable, "train_rift_dataset_pvc.py", *common, "--pvc-epochs", "2"], cwd=ROOT,
                         env=env, capture_output=True, text=True, timeout=600)
    assert cuda.returncode == 0, cuda.stderr[-1500:]
    assert pvc.returncode == 0, pvc.stderr[-1500:]
    cplan = json.loads(cuda.stdout[cuda.stdout.index("{"):])
    pplan = json.loads(pvc.stdout[pvc.stdout.index("{"):])
    assert pplan["entrypoint"] == "train_rift_dataset_pvc.py" and pplan["accelerator"] == "xpu"
    c, p = cplan["plans"][0], pplan["plans"][0]
    assert p["cuda_commands"] == c["commands"]
    ccmd, pcmd = c["commands"][0], p["commands"][0]
    # the planner prefixes the interpreter: [python, train.py, ...]
    k = 1 if ccmd[0] == sys.executable else 0
    # scripts are absolute paths under the project root in planned commands
    assert Path(ccmd[k]).name == "train.py" and Path(pcmd[k]).name == "train_pvc.py" and ccmd[:k] == pcmd[:k]
    assert Path(pcmd[k]).parent == Path(ccmd[k]).parent
    assert pcmd[pcmd.index("--epochs") + 1] == "2" and ccmd[ccmd.index("--epochs") + 1] == "150"
    assert pcmd[pcmd.index("--checkpoint-name") + 1] == "rift" + fe.SMOKE_SUFFIX
    # every other token identical
    strip = lambda cmd: [t for i, t in enumerate(cmd) if i == 0 or cmd[i - 1] not in ("--epochs", "--checkpoint-name", "--execution-contract-label")]
    assert strip(ccmd)[k + 1:] == strip(pcmd)[k + 1:]
    assert p["output_dir"] == c["output_dir"] + fe.SMOKE_SUFFIX
    assert p["role_manifest_path"] == c["role_manifest_path"]


def test_gotcha_pvc_frontend_lists_pvc_backends():
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    proc = subprocess.run([sys.executable, "train_gotcha_dataset_pvc.py", "--list"], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-1500:]
    registry = json.loads(proc.stdout[proc.stdout.index("{"):])
    assert registry["rift"]["module"] == "rift_pvc.gotcha_training" and registry["rift"]["builtin"]
    for method in ("spinr", "geraf", "sugavanam_ertin", "radar_fields", "radarsplat"):
        assert registry[method]["module"].endswith("_pvc")
    assert registry["sh_sas"]["status"] == "unavailable"
    assert registry["sh_sas"]["module"] == "train_sh_sas_pvc"
    assert registry["sh_sas"]["reason"] == (
        "No PVC entry point with an explicit native GOTCHA backend yet (see RIFT_PVC_Adaptation.md).")


@needs_data
@pytest.mark.parametrize("smoke", [False, True])
def test_sh_sas_plan_preserves_production_arguments(tmp_path, smoke):
    common = ["--dataset-root", DATA_ROOT, "--output-root", str(tmp_path / "out"),
              "--object", "b787", "--method", "sh_sas", "--num-train", "2400",
              "--num-tx", "1", "--num-rx", "1", "--dry-run"]
    old = fe.base.make_plan(fe.base.parse_args(common))
    new = fe.make_plan(fe.parse_args(common + (["--pvc-smoke"] if smoke else [])))
    c, p = old["plans"][0], new["plans"][0]
    expected = fe.to_pvc_command(c["commands"][0])
    if smoke:
        expected = fe.mark_smoke(expected)
    assert p["commands"] == [expected]
    assert p["output_dir"] == c["output_dir"] + (fe.SMOKE_SUFFIX if smoke else "")
    assert p["role_manifest_path"] == c["role_manifest_path"]
    assert "--device" not in expected
    assert expected[expected.index("--checkpoint-name") + 1] == "sh_sas" + (fe.SMOKE_SUFFIX if smoke else "")
    assert not (tmp_path / "out").exists()


def test_gotcha_frontend_file_on_disk_untouched():
    src = (ROOT / "train_gotcha_dataset.py").read_text()
    assert "rift_pvc" not in src and "_pvc" not in src


@pytest.mark.parametrize("failure", ["nonliteral", "empty", "list", "none", "syntax",
                                     "polarizations", "metric", "callable"])
def test_gotcha_invalid_unselected_backend_does_not_block_rift(tmp_path, monkeypatch, failure):
    import train_gotcha_dataset_pvc as cli
    spec = dict(schema=cli.BACKEND_SCHEMA, method="spinr", selection_unit="pass_sector",
                joint_passes=True, native_frequency_policy="ragged_exact",
                polarizations=["hh"], metric_domain="native_complex", callable="run_gotcha")
    if failure == "polarizations":
        spec["polarizations"] = [{"not": "a channel"}]
    elif failure == "metric":
        spec["metric_domain"] = ""
    elif failure == "callable":
        spec["callable"] = "missing"
    expression = {"nonliteral": "another.GOTCHA_BACKEND", "empty": "{}", "list": "[]",
                  "none": "None", "syntax": "{"}.get(failure, repr(spec))
    (tmp_path / "train_spinr_style_pvc.py").write_text(
        "GOTCHA_BACKEND = " + expression + "\ndef run_gotcha(): pass\n")
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    registry = cli.backend_registry()
    assert registry["spinr"]["status"] == "invalid"
    assert cli.resolve_methods(["rift"], registry, ["hh"]) == ["rift"]
    assert "spinr" not in cli.resolve_methods(["all"], registry, ["hh"])
    with pytest.raises(ValueError, match="spinr"):
        cli.resolve_methods(["spinr"], registry, ["hh"])


@needs_data
def test_dry_run_resume_is_forwarded_to_train_pvc(tmp_path):
    ckpt = tmp_path / "checkpoint_latest.pth.tar"
    ckpt.write_bytes(b"")
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    proc = subprocess.run([sys.executable, "train_rift_dataset_pvc.py", "--dataset-root", DATA_ROOT, "--output-root",
                           str(tmp_path / "out"), "--object", "b787", "--method", "rift", "--num-train", "2400",
                           "--num-tx", "1", "--num-rx", "1", "--dry-run", "--pvc-epochs", "3", "--resume", str(ckpt)],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-1500:]
    plan = json.loads(proc.stdout[proc.stdout.index("{"):])
    cmd = plan["plans"][0]["commands"][0]
    assert cmd[cmd.index("--resume") + 1] == str(ckpt.resolve()) and cmd[cmd.index("--epochs") + 1] == "3"


def test_gotcha_registry_reports_invalid_declaration_without_blocking_others(tmp_path, monkeypatch):
    """A malformed GOTCHA_BACKEND in an unselected `_pvc` module must not abort planning."""
    import train_gotcha_dataset_pvc as gfe
    bad = tmp_path / "train_spinr_style_pvc.py"
    bad.write_text("import train_spinr_style\nGOTCHA_BACKEND = train_spinr_style.GOTCHA_BACKEND\n")
    monkeypatch.setattr(gfe, "PROJECT_ROOT", tmp_path)
    registry = gfe.backend_registry()
    assert registry["spinr"]["status"] == "invalid" and "literal mapping" in registry["spinr"]["reason"]
    assert registry["rift"]["status"] == "available"
    assert gfe.resolve_methods(["rift"], registry, ["hh"]) == ["rift"]
    with pytest.raises(ValueError):
        gfe.resolve_methods(["spinr"], registry, ["hh"])
