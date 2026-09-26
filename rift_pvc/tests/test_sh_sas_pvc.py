"""Package H CLI, source isolation, sealed-role and continuation contracts."""
import ast
import inspect
import json
import os
import signal
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def twins(monkeypatch):
    from rift_pvc import sh_sas_training
    monkeypatch.setenv("RIFT_ACCELERATOR", "cpu")
    monkeypatch.setenv("RIFT_PVC_ALLOW_BACKEND", "cpu")
    return sh_sas_training


def make_sealed_fixture(root):
    root.mkdir(parents=True, exist_ok=True)
    # Two train views, one validation, one sealed test and one sealed unused.
    response = (np.random.default_rng(3).normal(size=(5, 1, 1, 1, 16))
                + 1j * np.random.default_rng(4).normal(size=(5, 1, 1, 1, 16))).astype(np.complex64)
    response[3:] = np.nan  # a response access guard below fails before these rows are read
    positions = np.array([[7, 3, 6], [3, 7, 6], [6, 3, 7], [-7, 3, 6], [7, -3, 6]], np.float32)
    path = root / "fixture.npz"
    np.savez(path, response=response, viewpoint_positions=positions,
             tx_pos=positions[:, None], rx_pos=positions[:, None] + 0.01,
             metadata_json=json.dumps({"radar_fc_hz": 1e10, "radar_bandwidth_hz": 3e9,
                                       "num_adc_samples": 16}))
    manifest = root / "roles.json"
    manifest.write_text(json.dumps({
        "schema_version": 1, "name": "sh_sas_pvc_synthetic",
        "dataset": {"num_views": 5, "response_shape": list(response.shape), "response_dtype": "complex64"},
        "split": {"strategy": "explicit", "complete_partition": True,
                  "test_sealed": True, "unused_sealed": True,
                  "num_train": 2, "num_validation": 1, "num_test": 1, "num_unused": 1,
                  "train_indices": [0, 1], "validation_indices": [2],
                  "test_indices": [3], "unused_indices": [4]}}))
    return ["--npz-path", str(path), "--npz-role-manifest", str(manifest),
            "--num-train", "2", "--num-val", "1", "--num-test", "1",
            "--checkpoint-root", str(root), "--steps", "12", "--eval-every", "4",
            "--checkpoint-every", "2", "--granularity", "4", "--hash-levels", "2",
            "--hash-log2-size", "6", "--hash-base-resolution", "4",
            "--hash-final-resolution", "8", "--hidden-dim", "8",
            "--query-chunk", "32", "--num-freq-wanted", "8", "--phase-sign", "-1"]


def assert_nested_close(a, b, *, rtol=0, atol=0):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a.cpu(), b.cpu(), rtol=rtol, atol=atol)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_nested_close(a[key], b[key], rtol=rtol, atol=atol)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_nested_close(x, y, rtol=rtol, atol=atol)
    elif isinstance(a, float):
        assert a == pytest.approx(b, rel=rtol, abs=atol)
    else:
        assert a == b


def run_continuation_case(root, monkeypatch, device, *, rtol=0, atol=0):
    import train_sh_sas as base
    import train_sh_sas_pvc as entry
    from rift.radar_fields_dataset import RadarFieldsArrays
    argv = make_sealed_fixture(root) + ["--device", device]
    reads = []
    original_read = RadarFieldsArrays.response_view

    def read(arrays, index):
        assert index in (0, 1, 2), "reserved response read"
        assert arrays.response_access_is_restricted
        reads.append(index)
        return original_read(arrays, index)

    monkeypatch.setattr(RadarFieldsArrays, "response_view", read)
    entry.main(argv + ["--checkpoint-name", "full"])
    objective = base.view_objective
    count = 0

    def interrupt(*args, **kwargs):
        nonlocal count
        result = objective(*args, **kwargs)
        if kwargs.get("include_regularizers"):
            count += 1
            if count == 5:
                os.kill(os.getpid(), signal.SIGTERM)
        return result

    with monkeypatch.context() as patch:
        patch.setattr(base, "view_objective", interrupt)
        entry.main(argv + ["--checkpoint-name", "resumed"])
    checkpoint = root / "resumed/checkpoint_latest.pth.tar"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert state["step"] == 5
    assert state["accelerator_backend"] == torch.device(device).type
    if device.startswith("xpu"):
        assert state["xpu_rng_state"]
        mapped = torch.load(checkpoint, map_location="xpu:0", weights_only=False)
        assert next(iter(mapped["sh_sas_state_dict"].values())).device.type == "xpu"
    entry.main(argv + ["--checkpoint-name", "resumed", "--resume", str(checkpoint)])
    full = torch.load(root / "full/checkpoint_final.pth.tar", map_location="cpu", weights_only=False)
    resumed = torch.load(root / "resumed/checkpoint_final.pth.tar", map_location="cpu", weights_only=False)
    for key in ("sh_sas_state_dict", "model_state_dict", "gain_state_dict", "optimizer_state_dict",
                "torch_rng_state", "numpy_rng_state_json", "sealed_npz_protocol_contract", "step"):
        assert_nested_close(full[key], resumed[key], rtol=rtol, atol=atol)
    if device == "cpu":
        # Wall clock is telemetry, not part of trajectory equivalence.
        history = lambda ck: [{k: v for k, v in row.items() if k != "step_seconds"} for row in ck["history"]]
        assert_nested_close(history(full), history(resumed))
    reads.clear()
    entry.main(argv + ["--checkpoint-name", "resumed", "--resume",
                       str(root / "resumed/checkpoint_final.pth.tar"), "--eval-only"])
    assert reads == [2]
    assert json.loads((root / "resumed/sh_sas_eval.json").read_text())["complete"] == 1
    return full, resumed


def test_interrupt_resume_and_validation_only(tmp_path, monkeypatch, twins, capsys):
    run_continuation_case(tmp_path, monkeypatch, "cpu")
    output = capsys.readouterr().out
    assert "Received signal 15" in output and "Stopped cleanly" in output
    assert "Resumed SH-SAS from step 5" in output


def test_parse_args_parity_and_explicit_device(twins):
    argv = ["--npz-path", "unused", "--views-per-step", "2", "--no-lambertian"]
    old = vars(twins.original_parse_args(argv))
    new = vars(twins.parse_args(argv))
    assert {k: v for k, v in old.items() if k != "device"} == {k: v for k, v in new.items() if k != "device"}
    assert new["device"] == "cpu"
    assert twins.parse_args(argv + ["--device=xpu:0"]).device == "xpu:0"


def test_install_is_exact_and_idempotent(twins):
    import train_sh_sas as base
    before = dict(vars(base))
    twins.install()
    expected = {"parse_args", "set_seed", "checkpoint_payload", "restore_device_rng_state", "SHSASField"}
    changed = {k for k, v in vars(base).items() if k not in before or before[k] is not v}
    assert changed <= expected
    for name in expected:
        assert getattr(base, name) is getattr(twins, name)
    once = dict(vars(base))
    twins.install()
    assert all(vars(base)[k] is v for k, v in once.items())


def test_main_copy_has_only_the_rng_edit(twins):
    import train_sh_sas_pvc as entry
    original = ast.parse((ROOT / "train_sh_sas.py").read_text())
    fn = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    fn.name = "_main"
    class Restore(ast.NodeTransformer):
        def visit_If(self, node):
            if ast.unparse(node.test) == "device.type == 'cuda' and checkpoint.get('cuda_rng_state') is not None":
                return ast.parse("restore_device_rng_state(checkpoint)").body[0]
            return self.generic_visit(node)
    expected = Restore().visit(fn)
    actual = ast.parse(inspect.getsource(entry._main)).body[0]
    assert ast.dump(actual) == ast.dump(expected)


def test_checkpoint_payload_parity(twins):
    import train_sh_sas as base
    from rift.calibration import GlobalComplexGain
    args = twins.parse_args(["--npz-path", "unused", "--granularity", "4",
                            "--hash-levels", "2", "--hash-log2-size", "5"])
    model = base.build_model(args, torch.device("cpu"))
    gain = GlobalComplexGain()
    optimizer = torch.optim.Adam([*model.parameters(), *gain.parameters()], eps=1e-15)
    inputs = (model, gain, optimizer, 5, 0.3, np.random.default_rng(42), [], args)
    old = twins.original_checkpoint_payload(*inputs)
    new = twins.checkpoint_payload(*inputs)
    assert new.keys() - old.keys() == {"xpu_rng_state", "accelerator_backend", "sh_sas_backend"}
    assert_nested_close(old, {k: new[k] for k in old})
    assert new["xpu_rng_state"] is None and new["accelerator_backend"] == "cpu"


@pytest.mark.parametrize("backend", ["cpu", "xpu", "cuda"])
def test_restore_selects_backend_and_tolerates_absence(twins, monkeypatch, backend, capsys):
    monkeypatch.setattr(twins.accelerator, "backend", lambda: backend)
    restored = []
    monkeypatch.setattr(twins.accelerator, "set_rng_state_all", restored.append)
    states = {"xpu_rng_state": [torch.tensor([1], dtype=torch.uint8)],
              "cuda_rng_state": [torch.tensor([2], dtype=torch.uint8)]}
    twins.restore_device_rng_state(states)
    if backend == "cpu":
        assert not restored
    else:
        assert_nested_close(restored[0], states[backend + "_rng_state"])
        twins.restore_device_rng_state({})
        assert "no " + backend + " RNG state" in capsys.readouterr().out


def test_backend_guard_checks_requested_device(twins, monkeypatch):
    monkeypatch.delenv("RIFT_PVC_ALLOW_BACKEND")
    with pytest.raises(RuntimeError, match="PVC"):
        twins.require_backend("cpu")
    monkeypatch.setenv("RIFT_PVC_ALLOW_BACKEND", "cpu")
    assert twins.require_backend("cpu") == "cpu"


def test_original_sources_have_no_pvc_imports():
    for name in ("train_sh_sas.py", "rift/sh_sas.py", "rift/radar_fields.py",
                 "rift/range_operator.py", "rift/occlusion.py"):
        assert "rift_pvc" not in (ROOT / name).read_text()


@pytest.mark.parametrize("failure", [None, "exit", "stalled", "fallback", "recipe", "roles", "command", "missing_stop"])
def test_smoke_report_rejects_failed_acceptance(tmp_path, twins, failure):
    from scripts_pvc.sh_sas_pvc_smoke_report import report
    args = vars(twins.parse_args(["--npz-path", "unused", "--num-train", "2400",
                                  "--num-val", "1000", "--num-test", "1000",
                                  "--num-freq-wanted", "600", "--phase-sign", "-1"]))
    checkpoint_path = str(tmp_path / "checkpoint_latest.pth.tar")
    command = ["python", "train_sh_sas_pvc.py", "--npz-path", "unused"]
    for number in (1, 2):
        stage = {"exit_code": 0, "checkpoint_path": checkpoint_path, "events": [],
                 "command": command + (["--resume", checkpoint_path] if number == 2 else []),
                 "checkpoint": {"step": number * 20, "accelerator_backend": "xpu", "xpu_rng_entries": 1,
                                "sh_sas_backend": "sh_sas_torch_xpu_tensor_divide_v1", "args": dict(args),
                                "history": [], "contract": {
                                    "dataset_identity": {"object_id": "b787"},
                                    "antenna_selection": {"tx_indices": [0], "rx_indices": [0]},
                                    "role_ids": {k: list(range(v)) for k, v in {
                                        "train": 2400, "validation": 1000, "reserved_test": 1000, "unused": 5600}.items()},
                                    "response_access": {"reserved_test_materialized": False,
                                                        "unused_materialized": False}}}}
        log = "Received signal 15\nStopped cleanly\nStep [20/1000] [1.2s/step]\n"
        if number == 2:
            log += "Resumed SH-SAS from step 20\n"
            if failure == "exit":
                stage["exit_code"] = 143
            elif failure == "stalled":
                stage["checkpoint"]["step"] = 20
            elif failure == "fallback":
                log += "Aten Op fallback from XPU to CPU\n"
            elif failure == "recipe":
                stage["checkpoint"]["args"]["steps"] = 20
            elif failure == "roles":
                stage["checkpoint"]["contract"]["response_access"]["reserved_test_materialized"] = True
            elif failure == "command":
                stage["command"] += ["--lr", "0.2"]
            elif failure == "missing_stop":
                log = "Step [40/1000] [1.2s/step]\nResumed SH-SAS from step 20\n"
        (tmp_path / f"phase{number}.json").write_text(json.dumps(stage))
        (tmp_path / f"phase{number}.log").write_text(log)
    result = report(tmp_path)
    assert result["passed"] == (failure is None)
    assert result["validation_sizing_available"] is False


def test_query_copy_changes_only_xpu_divisor():
    from rift.sh_sas import SHSASField as Original
    from rift_pvc.sh_sas import SHSASField as PVC
    import textwrap
    original = ast.parse(textwrap.dedent(inspect.getsource(Original.query_coefficients))).body[0]
    adapted = ast.parse(textwrap.dedent(inspect.getsource(PVC.query_coefficients))).body[0]
    # Drop the docstring and the CPU delegation branch, then normalize the
    # single scalar-tensor divisor in the XPU copy.
    adapted.body = adapted.body[2:]
    original.body = original.body[1:]
    class Divisor(ast.NodeTransformer):
        def visit_Call(self, node):
            if ast.unparse(node) == "xyz.new_tensor(self.extent)":
                return ast.parse("self.extent", mode="eval").body
            return self.generic_visit(node)
    adapted = Divisor().visit(adapted)
    assert ast.dump(original) == ast.dump(adapted)
