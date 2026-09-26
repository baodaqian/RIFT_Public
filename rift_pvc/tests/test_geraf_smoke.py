"""Recovery-controller regressions: real process signals and synthetic training."""
import json
import os
from pathlib import Path
import sys
import textwrap

import pytest
import torch

from scripts_pvc.smoke_geraf_pvc import load_partial, recover, run_segment


@pytest.fixture(autouse=True)
def subprocess_repository_path(monkeypatch):
    """Temporary child scripts need the same import root as the test process."""
    root = str(Path(__file__).resolve().parents[2])
    inherited = os.environ.get("PYTHONPATH")
    monkeypatch.setenv("PYTHONPATH", root + (os.pathsep + inherited if inherited else ""))


def child(tmp_path, source):
    path = tmp_path/"trainer.py"
    path.write_text(textwrap.dedent(source))
    return [sys.executable, "-u", str(path)]


def test_real_training_resumes_without_compressing_production_clock(tmp_path, monkeypatch):
    monkeypatch.setenv("RIFT_ACCELERATOR", "cpu")
    command = child(tmp_path, '''
        import argparse, json
        from rift_pvc.geraf_source_training import train
        from tests.test_geraf_source import SMALL, TinyData
        p = argparse.ArgumentParser()
        p.add_argument('--checkpoint-dir', required=True)
        p.add_argument('--resume', action=argparse.BooleanOptionalAction, default=False)
        p.add_argument('--resume-path')
        a = p.parse_args()
        config = dict(SMALL, steps=50000, log_every=1, checkpoint_every=100, validation_every=1000)
        result = train(data=TinyData(), output_dir=a.checkpoint_dir, config=config,
                       device='cpu', resume=a.resume_path if a.resume else None)
        print(json.dumps(result), flush=True)
        raise SystemExit(143 if result['status'] == 'interrupted' else 0)
    ''') + ["--checkpoint-dir", str(tmp_path/"checkpoints"), "--no-resume"]
    report = recover(command, tmp_path, first_update=100, resume_updates=100, deadline=90, backend="cpu")
    assert report["status"] == "passed"
    assert report["final_step"] >= report["first_step"]+100
    assert report["production_steps"] == 50000
    saved = load_partial(report["checkpoint"], backend="cpu")
    from rift.vendor.geraf_sens.scheduler import IterLRScheduler
    optimizer = torch.optim.AdamW([
        {"params": [torch.nn.Parameter(torch.zeros(1))], "lr": 1e-4},
        {"params": [torch.nn.Parameter(torch.zeros(1))], "lr": 1e-3}])
    scheduler = IterLRScheduler(optimizer, [dict(type="CosineAnnealingLR", eta_min=5e-4)], 50000)
    scheduler.step(saved["step"])
    assert report["final_learning_rates"] == [g["lr"] for g in optimizer.param_groups]
    # A resumed smoke never pretends the production fitting budget is complete.
    assert saved["complete"] is False


@pytest.mark.parametrize("code", [0, 1, 42, 143])
def test_unrequested_exit_never_counts_as_recovery(tmp_path, code):
    command = child(tmp_path, f"raise SystemExit({code})")
    with pytest.raises(RuntimeError, match="GeRaF exit"):
        run_segment(command, tmp_path/"run.log", stop_step=1, deadline=5)


def test_fallback_warning_is_a_failure_even_with_exit_zero(tmp_path):
    command = child(tmp_path, "print('Aten Op fallback from XPU to CPU', flush=True)")
    with pytest.raises(RuntimeError, match="operator fallback"):
        run_segment(command, tmp_path/"run.log", deadline=5)


def test_watchdog_is_failure_even_after_cooperative_exit(tmp_path):
    command = child(tmp_path, '''
        import signal, time
        signal.signal(signal.SIGTERM, lambda *_: exit(143))
        time.sleep(30)
    ''')
    with pytest.raises(TimeoutError):
        run_segment(command, tmp_path/"run.log", stop_step=1, deadline=.5, grace=2)


def test_missing_partial_is_not_a_fresh_resume(tmp_path):
    with pytest.raises(RuntimeError, match="missing"):
        load_partial(tmp_path/"missing.pth.tar")


def test_missing_prepared_status_fails_even_on_exit_zero(tmp_path):
    command = child(tmp_path, "print('no work was performed')")
    with pytest.raises(RuntimeError, match="did not report prepared"):
        run_segment(command, tmp_path/"run.log", deadline=5)


def test_public_smoke_rejects_the_historical_shortened_schedule_before_launch(tmp_path, monkeypatch):
    from scripts_pvc import smoke_geraf_pvc as controller
    from rift_pvc import accelerator
    config = tmp_path/"historical.json"
    config.write_text(json.dumps(dict(mf_grid=48, bank_size=1, steps=24)))
    plan = tmp_path/"plan.json"
    plan.write_text(json.dumps({"command": ["python", "unused", "--source-config", str(config)]}))
    monkeypatch.setenv("SLURM_JOB_ID", "synthetic")
    monkeypatch.setattr(accelerator, "backend", lambda: "xpu")
    monkeypatch.setattr(accelerator, "device_count", lambda: 1)
    monkeypatch.setattr(sys, "argv", ["smoke_geraf_pvc.py", "--plan", str(plan)])
    with pytest.raises(ValueError, match="unchanged production"):
        controller.main()
    assert not list(tmp_path.glob("*.log"))


def test_provenance_uses_repository_commit_and_api_without_source_hashes():
    from rift_pvc import geraf_source
    from rift_pvc.vendor.geraf_sens.rf_rendering import GeRaFStage1
    root = Path(__file__).resolve().parents[2]
    notice = (root/"rift_pvc/vendor/geraf_sens/NOTICE.md").read_text()
    assert "https://github.com/VictorLlu/GeRaF-SENS/" in notice
    assert geraf_source.COMMIT in notice
    assert (root/"rift_pvc/vendor/geraf_sens/LICENSE").is_file()
    assert callable(GeRaFStage1.loss) and callable(geraf_source.predict_native)


def test_source_policy_deselects_only_the_historical_inventory_gate(pytester):
    directory = pytester.path/"tests"
    directory.mkdir()
    (directory/"test_geraf_source.py").write_text(textwrap.dedent('''
        def test_vendored_definitions_match_pinned_source_ast():
            raise AssertionError("Historical source gate must never execute")
        def test_behavior():
            assert True
    '''))
    result = pytester.runpytest_subprocess("-p", "rift_pvc.pytest_policy", "-q")
    result.assert_outcomes(passed=1, deselected=1)


pytest_plugins = ["pytester"]
