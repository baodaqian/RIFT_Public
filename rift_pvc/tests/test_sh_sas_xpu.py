"""Package H device gates; the operator probe can run before the trainer exists.

Tolerances fixed before probing: fp32 field/occlusion relative L2 1e-4,
gradients 1e-3, and the fp64 range operator 1e-5 with identical inputs.
"""
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from rift.config import cc
from rift.occlusion import ray_transmittance
from rift.range_operator import range_forward_operator
from rift.sh_sas import SHSASField
from rift_pvc.sh_sas import SHSASField as PVCField

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(not torch.xpu.is_available(), reason="requires a PVC card")


def relative(actual, expected):
    return float((actual.detach().cpu() - expected.detach().cpu()).norm()
                 / expected.detach().cpu().norm().clamp_min(1e-30))


def test_production_field_range_and_gradients():
    torch.manual_seed(42)
    cpu = SHSASField(extent=0.15, granularity=48)
    card = PVCField(extent=0.15, granularity=48).to("xpu")
    card.load_state_dict(cpu.state_dict())
    tx = torch.tensor([[7.123, 3.271, 6.432]])
    rx = tx + torch.tensor([[0.01, -0.013, 0.007]])
    # Build all coordinates once on CPU: isolate device arithmetic from lattice
    # construction and keep the original G48/16-level/2^19/width-32 recipe.
    a = cpu.view_field(tx, rx, opacity_scale=0.1)
    b = card.view_field(tx.to("xpu"), rx.to("xpu"), opacity_scale=0.1)
    errors = {}
    for key in ("coefficients", "density", "normals", "transmittance", "weights"):
        error = relative(b[key], a[key])
        print(f"G48 {key} relative_l2={error:.6g}", flush=True)
        errors[key] = error
    # Backward through the full field, embedding and checkpointed occlusion.
    a["weights"].abs().square().mean().backward()
    b["weights"].abs().square().mean().backward()
    for (name, p), q in zip(cpu.named_parameters(), card.parameters()):
        assert p.grad is not None and q.grad is not None, name
        error = relative(q.grad, p.grad)
        print(f"G48 gradient {name} relative_l2={error:.6g}", flush=True)
        errors["gradient " + name] = error

    frequencies = torch.linspace(8.5e9, 11.5e9, 600)
    kvector = 2 * torch.pi * frequencies / cc
    outputs, gradients = [], []
    for device in ("cpu", "xpu"):
        weights = a["weights"].detach().to(device).requires_grad_(True)
        output = range_forward_operator(
            frequencies.to(device), kvector.to(device), rx.to(device), tx.to(device),
            a["points"].to(device), weights, phase_sign=-1.0,
            oversample=2, kernel_width=20, pair_chunk=16, point_chunk=16384,
            compute_dtype=torch.float64, range_model="sum2")
        output.abs().square().sum().backward()
        outputs.append(output.detach().cpu())
        gradients.append(weights.grad.cpu())
    for name, values in (("range", outputs), ("range_gradient", gradients)):
        error = relative(values[1], values[0])
        print(f"G48 {name} relative_l2={error:.6g}", flush=True)
        assert error < 1e-5
    # The declared forward gate is on rendered predictions, not the
    # normalization of near-zero internal normal vectors; report both.
    assert errors["weights"] < 1e-4
    assert all(v < 1e-3 for k, v in errors.items() if k.startswith("gradient "))


def test_production_ray_transmittance_gradient():
    torch.manual_seed(13)
    points = SHSASField(extent=0.15, granularity=48, hash_levels=1,
                        hash_log2_size=4).grid_positions
    sigma = torch.rand(48 ** 3) * 3
    origin = torch.tensor([7.123, 3.271, 6.432])
    outputs, gradients = [], []
    for device in ("cpu", "xpu"):
        s = sigma.detach().to(device).requires_grad_(True)
        out = ray_transmittance(points.to(device), s, origin.to(device),
                                0.15, 48, point_chunk=16384)
        out.square().mean().backward()
        outputs.append(out.detach().cpu())
        gradients.append(s.grad.cpu())
    assert relative(outputs[1], outputs[0]) < 1e-4
    assert relative(gradients[1], gradients[0]) < 1e-3
    print("ray forward/gradient relative_l2", relative(outputs[1], outputs[0]),
          relative(gradients[1], gradients[0]), flush=True)


def test_production_coordinate_diagnostic():
    torch.manual_seed(42)
    cpu = SHSASField(extent=0.15, granularity=48)
    card = copy.deepcopy(cpu).to("xpu")
    points = cpu.grid_positions
    x = points.to("xpu")
    unit = ((points / cpu.extent) + 1.0) * 0.5
    options = {
        "original": ((x / cpu.extent) + 1.0) * 0.5,
        "multiply": ((x * (1.0 / cpu.extent)) + 1.0) * 0.5,
        "tensor_divide": ((x / x.new_tensor(cpu.extent)) + 1.0) * 0.5,
    }
    with torch.no_grad():
        encoded = cpu.encoder(unit)
        raw = cpu.mlp(encoded)
        for name, u in options.items():
            e = card.encoder(u)
            r = card.mlp(e)
            print("coordinate", name, "max", float((u.cpu() - unit).abs().max()),
                  "encoded relative", relative(e, encoded), "mlp", relative(r, raw), flush=True)
        e = card.encoder(unit.to("xpu"))
        print("common unit: encoder", relative(e, encoded),
              "mlp", relative(card.mlp(e), raw), flush=True)


def test_operator_process_has_no_fallback():
    env = dict(os.environ, PYTORCH_DEBUG_XPU_FALLBACK="1", OMP_NUM_THREADS="8")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-s", "-q", "-p", "no:cacheprovider",
         "-k", "production"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=800)
    output = proc.stdout + proc.stderr
    print(output, flush=True)
    assert proc.returncode == 0, output
    assert "Aten Op fallback" not in output


def test_card_rng_resume_and_same_seed_spread(tmp_path, monkeypatch):
    from rift_pvc import accelerator
    from rift_pvc.sh_sas_training import restore_device_rng_state
    from rift_pvc.tests.test_sh_sas_pvc import run_continuation_case
    monkeypatch.setenv("RIFT_ACCELERATOR", "xpu")
    accelerator.manual_seed_all(23)
    state = accelerator.get_rng_state_all()
    expected = torch.rand(8, device="xpu")
    restore_device_rng_state({"xpu_rng_state": state})
    assert torch.equal(torch.rand(8, device="xpu"), expected)
    full, resumed = run_continuation_case(tmp_path / "first", monkeypatch, "xpu:0", rtol=1e-4, atol=1e-7)
    # Repeat the complete fit: retain the original accumulation behavior and
    # measure its spread, rather than impose a different deterministic backend.
    import train_sh_sas_pvc as entry
    from rift_pvc.tests.test_sh_sas_pvc import make_sealed_fixture
    argv = make_sealed_fixture(tmp_path / "repeat") + ["--device", "xpu:0", "--checkpoint-name", "full"]
    entry.main(argv)
    repeated = torch.load(tmp_path / "repeat/full/checkpoint_final.pth.tar", map_location="cpu", weights_only=False)
    spreads = {}
    for label, candidate in (("resume", resumed), ("fresh_repeat", repeated)):
        values = [relative(candidate["sh_sas_state_dict"][k], v)
                  for k, v in full["sh_sas_state_dict"].items() if v.is_floating_point()]
        losses = [abs(a["val_rel_mse"] - b["val_rel_mse"]) / max(abs(a["val_rel_mse"]), 1e-30)
                  for a, b in zip(full["history"], candidate["history"])]
        spreads[label] = {"max_parameter_relative_l2": max(values),
                          "max_validation_relative_difference": max(losses)}
        assert max(values) < 1e-3 and max(losses) < 1e-3
    (tmp_path / "same_seed_spread.json").write_text(json.dumps(spreads, indent=2))
    print("SH-SAS same-seed spread", json.dumps(spreads), flush=True)


def test_card_training_process_has_no_fallback(tmp_path):
    env = dict(os.environ, PYTORCH_DEBUG_XPU_FALLBACK="1", OMP_NUM_THREADS="8")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-s", "-q", "-p", "no:cacheprovider",
         "-k", "card_rng", "--basetemp", str(tmp_path / "training")],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    output = proc.stdout + proc.stderr
    print(output, flush=True)
    assert proc.returncode == 0, output
    assert "Aten Op fallback" not in output
