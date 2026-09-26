"""Regression checks for PVC recovery settings and smoke acceptance failures."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rift_pvc import tcnn_torch
from rift_pvc.radar_fields_training import validate_shim_identity
from scripts_pvc import radar_fields_smoke_report as smoke


@pytest.fixture(autouse=True)
def defaults(monkeypatch):
    monkeypatch.delenv("RIFT_PVC_TCNN_HALF", raising=False)
    monkeypatch.delenv("RIFT_PVC_TCNN_WEIGHT_GRAD", raising=False)


def test_existing_runtime_metadata_remains_compatible_but_changed_reduction_is_refused(monkeypatch):
    args = SimpleNamespace(model_backend="upstream-tcnn-torchshim")
    saved = {"tcnn_shim": tcnn_torch.identity()}
    saved["tcnn_shim"]["parity"] = {"status": "pending"}  # historical checkpoint
    validate_shim_identity(saved, args)
    monkeypatch.setenv("RIFT_PVC_TCNN_WEIGHT_GRAD", "gemm")
    with pytest.raises(ValueError, match="weight_grad"):
        validate_shim_identity(saved, args)
    monkeypatch.setenv("RIFT_PVC_TCNN_WEIGHT_GRAD", "bmm:64")
    with pytest.raises(ValueError, match="weight_grad"):
        validate_shim_identity(saved, args)


@pytest.mark.parametrize("field", ["model_backend", "version", "precision", "weight_grad"])
def test_incomplete_runtime_identity_is_not_guessed(field):
    saved = {"tcnn_shim": tcnn_torch.identity()}
    del saved["tcnn_shim"][field]
    with pytest.raises(ValueError, match=field):
        validate_shim_identity(saved, SimpleNamespace(model_backend="upstream-tcnn-torchshim"))


def checkpoint(root, dataset, step=1, complete=False):
    root.mkdir(exist_ok=True)
    controls = dict(model_backend="upstream-tcnn-torchshim", steps=4)
    opt = {"state": {0: {"step": torch.tensor(float(step)), "exp_avg": torch.ones(2)}}, "param_groups": []}
    rng = torch.zeros(16, dtype=torch.uint8)
    ck = dict(step=step, accelerator_backend="xpu", tcnn_shim=tcnn_torch.identity(),
              history=[{"step": step}], training_view_coverage={})
    if dataset == "rift":
        ck.update(artifact_schema="radar_fields_checkpoint_v2", args=controls, radar_fields_recipe={"model_backend": controls["model_backend"]},
                  radar_fields_state_dict={"p": torch.ones(2)}, optimizer_state_dict=opt,
                  scheduler_state_dict={"last_epoch": step}, xpu_rng_state=[rng], torch_rng_state=rng,
                  numpy_rng_state_json='{"state": 1}', dataset_provenance={"scene": "synthetic"},
                  split_provenance={"train": [0]}, sealed_protocol_contract={"test": "sealed"})
        name = "checkpoint_final.pth.tar" if complete else "checkpoint_latest.pth.tar"
    else:
        ck.update(schema="rift_gotcha_checkpoint_v1", recipe={"method": "radar_fields", "controls": controls},
                  model_state_dict={"p": torch.ones(2)}, optimizer=opt, scheduler={"last_epoch": step},
                  rng_xpu=[rng], rng_torch=rng, rng_numpy={"state": 1}, rng_python=(3, (1,), None),
                  dataset_contract={"test": "sealed"}, dataset_identity="synthetic", complete=complete, pending_validation=None)
        name = "checkpoint_final.pt" if complete else "checkpoint_latest.pt"
    path = root / name
    torch.save(ck, path)
    return path, ck


def trainer_log(path, dataset, step=1, complete=False, resume=None):
    if dataset == "rift":
        text = f"Resumed Radar Fields from step {resume}\n" if resume is not None else ""
        text += f"Step [{step}/4] loss=1\n"
        if not complete:
            text += "Received signal 15\nStopped cleanly after publishing checkpoint_latest.\n"
    else:
        text = json.dumps({"method": "radar_fields", "result": {
            "status": "complete" if complete else "interrupted", "step": step, "test_accessed": False}})
    path.write_text(text)
    return 0 if dataset == "rift" or complete else 143


@pytest.mark.parametrize("dataset", ["rift", "gotcha"])
def test_accepts_cooperative_progress_and_verified_completion(tmp_path, dataset):
    root, log = tmp_path / "run", tmp_path / "trainer.log"
    checkpoint(root, dataset)
    rc = trainer_log(log, dataset)
    first = smoke.report(root, dataset, log, rc)
    assert first["status"] == "passed" and not first["recovery_exercised"]
    with pytest.raises(ValueError, match="advance"):
        smoke.report(root, dataset, log, rc, first)
    checkpoint(root, dataset, step=2)
    rc = trainer_log(log, dataset, step=2, resume=1)
    second = smoke.report(root, dataset, log, rc, first)
    assert second["recovery_exercised"] and second["step"] == 2
    checkpoint(root, dataset, step=4, complete=True)
    rc = trainer_log(log, dataset, step=4, complete=True, resume=2)
    assert smoke.report(root, dataset, log, rc, second)["complete"]


@pytest.mark.parametrize("dataset", ["rift", "gotcha"])
@pytest.mark.parametrize("defect", ["fallback", "timeout", "crash", "missing_rng", "missing_optimizer", "nonfinite", "bad_clock", "missing_checkpoint", "empty_log", "false_completion"])
def test_smoke_rejects_invalid_execution_evidence(tmp_path, dataset, defect):
    root, log = tmp_path / "run", tmp_path / "trainer.log"
    path, ck = checkpoint(root, dataset)
    rc = trainer_log(log, dataset)
    if defect == "fallback":
        log.write_text(log.read_text() + "\nAten Op fallback from XPU to CPU\n")
    elif defect == "timeout":
        rc = 124
    elif defect == "crash":
        rc = 42
    elif defect == "empty_log":
        log.write_text("")
    elif defect == "missing_checkpoint":
        path.unlink()
    else:
        if defect == "missing_rng": ck["xpu_rng_state" if dataset == "rift" else "rng_xpu"] = []
        if defect == "missing_optimizer": ck["optimizer_state_dict" if dataset == "rift" else "optimizer"] = None
        if defect == "nonfinite": ck["radar_fields_state_dict" if dataset == "rift" else "model_state_dict"]["p"][0] = float("nan")
        if defect == "bad_clock": ck["scheduler_state_dict" if dataset == "rift" else "scheduler"]["last_epoch"] = 0
        if defect == "false_completion":
            ck["step"] = 4
            ck["scheduler_state_dict" if dataset == "rift" else "scheduler"]["last_epoch"] = 4
            ck["history"] = []
            ck["complete"] = True
            rc = trainer_log(log, dataset, step=4, complete=True)
            if dataset == "rift":
                path.unlink()
                path = root / "checkpoint_final.pth.tar"
        torch.save(ck, path)
    with pytest.raises((ValueError, KeyError)):
        smoke.report(root, dataset, log, rc)


def test_changed_runtime_is_rejected_before_collection_responses(monkeypatch, tmp_path):
    import train_radar_fields_pvc as entry
    from rift_pvc import radar_fields_training as twins
    monkeypatch.setenv("RIFT_ACCELERATOR", "cpu")
    monkeypatch.setenv("RIFT_PVC_ALLOW_BACKEND", "cpu")
    argv = ["--recipe", "source-adapted-v3", "--npz-path", "absent.npz", "--device", "cpu",
            "--sealed-protocol", "--sealed-split-manifest", "absent.json", "--resume", "unused.pt",
            "--num-train", "2400", "--num-val", "1000", "--num-test", "1000"]
    args = entry.parse_args(argv)
    saved = {"args": vars(args), "radar_fields_recipe": twins.recipe_contract(args), "tcnn_shim": tcnn_torch.identity()}
    monkeypatch.setattr(torch, "load", lambda *a, **kw: copy.deepcopy(saved))
    monkeypatch.setenv("RIFT_PVC_TCNN_WEIGHT_GRAD", "gemm")
    with pytest.raises(ValueError, match="weight_grad"):
        entry.main(argv)  # would fail on absent dataset paths if response access came first
