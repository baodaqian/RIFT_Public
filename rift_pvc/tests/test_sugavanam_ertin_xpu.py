"""Package C (Sugavanam--Ertin): the adapted code running on the accelerator.

Skipped entirely when no CUDA/XPU device is visible, so the file is safe in the
CPU test sweep. On a PVC card these cover the three things the CPU tests
cannot: that the production numerics have no XPU->CPU operator fallback, that
fp64/complex128 results agree with CPU, and that the workflow runs end to end
on the device and resumes exactly.

Run on a card with:
    sbatch --export=ALL,PYTEST_TARGET=rift_pvc/tests/test_sugavanam_ertin_xpu.py \\
        scripts_pvc/run_tests_pvc.sbatch
"""
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rift.sugavanam_ertin_acquisition import fourier_forward  # noqa: E402
from rift.sugavanam_ertin_paper import PaperSDF, field_gradient, sdf_losses  # noqa: E402
from rift.sugavanam_ertin_sparse import project_complex_l1_ball  # noqa: E402
from rift_pvc import accelerator  # noqa: E402
from rift_pvc import sugavanam_ertin_paper_workflow as workflow_pvc  # noqa: E402

pytestmark = pytest.mark.skipif(
    not accelerator.is_available(), reason="no CUDA/XPU device in this process")


def _load_original_tests():
    """Import tests/test_sugavanam_ertin_paper.py (not a package) for its fixture."""
    path = ROOT / "tests" / "test_sugavanam_ertin_paper.py"
    spec = importlib.util.spec_from_file_location("_se_paper_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def original_tests():
    return _load_original_tests()


@pytest.fixture
def device():
    return accelerator.device()


class _FallbackWatch:
    """Fail on any 'Aten Op fallback from XPU to CPU' warning (dispatch rule 5)."""

    def __enter__(self):
        self._ctx = warnings.catch_warnings(record=True)
        self.records = self._ctx.__enter__()
        warnings.simplefilter("always")
        return self

    def __exit__(self, *exc):
        self._ctx.__exit__(*exc)
        return False

    def assert_clean(self):
        hits = [str(r.message) for r in self.records
                if "fallback" in str(r.message).lower()]
        assert not hits, f"XPU->CPU operator fallback detected: {hits}"


# --------------------------------------------------------------------------
# Device numerics used by the production lane (audit C.2)
# --------------------------------------------------------------------------

def test_fourier_forward_matches_cpu_without_fallback(device):
    torch.manual_seed(0)
    points_cpu = torch.randn(64, 3, dtype=torch.float64) * 0.05
    weights_cpu = torch.randn(64, dtype=torch.complex128)
    frequencies = np.linspace(1e10, 1.3e10, 24)
    directions_cpu = torch.tensor([[1.0, 0.1, 0.0], [-1.0, -0.2, 0.0]], dtype=torch.float64)
    reference_cpu = torch.tensor([10.0, 10.5], dtype=torch.float64)
    amplitude_cpu = torch.ones(2, dtype=torch.float64)
    kwargs = dict(cc=299792458.0, point_chunk=32, pair_chunk=1)

    expected = fourier_forward(points_cpu, weights_cpu, frequencies, directions_cpu,
                               reference_cpu, amplitude_cpu, **kwargs)
    with _FallbackWatch() as watch:
        got = fourier_forward(points_cpu.to(device), weights_cpu.to(device), frequencies,
                              directions_cpu.to(device), reference_cpu.to(device),
                              amplitude_cpu.to(device), **kwargs)
        accelerator.synchronize()
    watch.assert_clean()
    assert got.dtype == torch.complex128 and got.device.type == device.type
    torch.testing.assert_close(got.cpu(), expected, rtol=1e-10, atol=1e-10)


def test_fourier_forward_gradient_through_checkpoint_matches_cpu(device):
    torch.manual_seed(1)
    points = torch.randn(32, 3, dtype=torch.float64) * 0.05
    frequencies = np.linspace(1e10, 1.1e10, 16)
    directions = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    reference = torch.tensor([10.0], dtype=torch.float64)
    amplitude = torch.ones(1, dtype=torch.float64)
    kwargs = dict(cc=299792458.0, point_chunk=8, pair_chunk=1)

    def grad_for(dev):
        w = torch.randn(32, dtype=torch.complex128, generator=torch.Generator().manual_seed(2))
        w = w.to(dev).requires_grad_(True)
        out = fourier_forward(points.to(dev), w, frequencies, directions.to(dev),
                              reference.to(dev), amplitude.to(dev), **kwargs)
        loss = 0.5 * out.abs().square().sum()
        return torch.autograd.grad(loss, w)[0].detach().cpu()

    expected = grad_for(torch.device("cpu"))
    with _FallbackWatch() as watch:
        got = grad_for(device)
        accelerator.synchronize()
    watch.assert_clean()
    torch.testing.assert_close(got, expected, rtol=1e-9, atol=1e-9)


def test_complex_l1_projection_matches_cpu_without_fallback(device):
    torch.manual_seed(3)
    x = torch.randn(512, dtype=torch.complex128)
    radius = float(x.abs().sum()) * 0.25
    expected = project_complex_l1_ball(x, radius)
    with _FallbackWatch() as watch:
        got = project_complex_l1_ball(x.to(device), radius)
        accelerator.synchronize()
    watch.assert_clean()
    torch.testing.assert_close(got.cpu(), expected, rtol=1e-12, atol=1e-12)
    assert float(got.abs().sum()) <= radius * (1 + 1e-9)


def test_historical_stage2_cpu_draws_and_refresh_run_on_device(device):
    from rift_pvc.sugavanam_ertin_stage2_sampling import CpuGeneratorTorch, sample_roi
    from rift_pvc.sugavanam_ertin_stage2_refresh import refresh_iso_points_strict
    from rift.sugavanam_ertin_a320_stabilized import refresh_iso_points_strict as original
    proxy = CpuGeneratorTorch()
    generator = proxy.Generator("xpu").manual_seed(17)
    reference = torch.Generator().manual_seed(17)
    assert generator.device.type == "cpu"
    with _FallbackWatch() as watch:
        for name, shape in (("rand", (7, 3)), ("randn", (7, 3)),
                            ("randint", (10, (7,))), ("randperm", (11,))):
            got = getattr(proxy, name)(*shape, generator=generator, device=device)
            expected = getattr(torch, name)(*shape, generator=reference)
            assert got.device.type == device.type
            torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(sample_roi(7, .15, generator, device).cpu(),
                                   sample_roi(7, .15, reference, "cpu"), rtol=0, atol=0)
        accelerator.synchronize()
    watch.assert_clean()
    assert torch.equal(generator.get_state(), reference.get_state())

    class Sphere(torch.nn.Module):
        def forward(self, points):
            return points.norm(dim=-1) - .05

    seeds = torch.randn(128, 3, generator=torch.Generator().manual_seed(1))
    seeds = torch.nn.functional.normalize(seeds, dim=-1) * .05
    g_cpu, g_device = torch.Generator().manual_seed(2), proxy.Generator("xpu").manual_seed(2)
    expected, expected_audit = original(Sphere(), seeds, .15, .002, 32, g_cpu)
    with _FallbackWatch() as watch:
        got, audit = refresh_iso_points_strict(Sphere(), seeds.to(device), .15, .002, 32, g_device)
        accelerator.synchronize()
    watch.assert_clean()
    assert got.device.type == device.type
    assert audit["status"] == expected_audit["status"] == "passed"
    assert audit["min_acceptance"] == .6
    torch.testing.assert_close(got.cpu(), expected, rtol=1e-5, atol=1e-7)
    assert torch.equal(g_cpu.get_state(), g_device.get_state())


def test_sdf_forward_backward_and_losses_without_fallback(device):
    recipe = workflow_pvc.make_recipe("rift_collection", {"initialization_std": 0.05})
    model = PaperSDF(**workflow_pvc._model_config(recipe, 0.15)).to(device)
    g = torch.Generator().manual_seed(4)
    on = (torch.rand(128, 3, generator=g) * 2 - 1) * 0.15
    normals = torch.nn.functional.normalize(torch.randn(128, 3, generator=g), dim=-1)
    off = (torch.rand(128, 3, generator=g) * 2 - 1) * 0.15
    with _FallbackWatch() as watch:
        losses = sdf_losses(model, on.to(device), normals.to(device), off.to(device),
                            None, None, extent=0.15, alpha_off=recipe["alpha_off"])
        total = sum(recipe["lambda_" + k] * v for k, v in losses.items())
        total.backward()
        accelerator.synchronize()
    watch.assert_clean()
    assert torch.isfinite(total).item()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


# --------------------------------------------------------------------------
# The published-initialization gate on the device (audit C.0 finding 2)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("std,expected", [(1.0, "initialization_degenerate"),
                                          (0.05, "initialization_probe_passed")])
def test_initialization_gate_on_device_agrees_with_cpu(device, std, expected):
    """The gate must fire identically on XPU; std=0.05 is the production value."""
    recipe = workflow_pvc.make_recipe("rift_collection", {"initialization_std": std})
    probes = (torch.rand(256, 3, generator=torch.Generator().manual_seed(42)) * 2 - 1) * 0.15
    results = {}
    for dev in (torch.device("cpu"), device):
        model = PaperSDF(**workflow_pvc._model_config(recipe, 0.15)).to(dev)
        with _FallbackWatch() as watch:
            f, grad = field_gradient(model, probes.to(dev))
            accelerator.synchronize()
        if dev.type != "cpu":
            watch.assert_clean()
        norms = grad.detach().norm(dim=-1)
        results[dev.type] = dict(
            saturated=float((f.detach().abs() == 1).float().mean()),
            nonzero_gradients=int((norms > 0).sum()),
            degenerate=not bool((norms > 0).any()))
    assert results["cpu"] == results[device.type], results
    degenerate = results[device.type]["degenerate"]
    assert degenerate == (expected == "initialization_degenerate")
    if degenerate:
        assert results[device.type]["saturated"] == 1.0
        assert results[device.type]["nonzero_gradients"] == 0


# --------------------------------------------------------------------------
# End-to-end on the device, with resume (audit C.1)
# --------------------------------------------------------------------------

def test_workflow_runs_end_to_end_on_device_and_resumes_exactly(tmp_path, device, original_tests):
    recipe = original_tests.tiny_recipe()
    continuous, interrupted = tmp_path / "continuous", tmp_path / "interrupted"
    with _FallbackWatch() as watch:
        workflow_pvc.run(original_tests.TinyAcquisition(), recipe, continuous, device=device)
        accelerator.synchronize()
    watch.assert_clean()

    calls = []

    def stop():
        calls.append(1)
        return len(calls) == 7

    result = workflow_pvc.run(original_tests.TinyAcquisition(), recipe, interrupted,
                              device=device, should_stop=stop)
    assert result["status"] == "interrupted"
    workflow_pvc.run(original_tests.TinyAcquisition(), recipe, interrupted, device=device,
                     resume=interrupted / "checkpoint_latest.pt")
    a = original_tests.load(continuous / "checkpoint_final.pt")
    b = original_tests.load(interrupted / "checkpoint_final.pt")
    original_tests.compare_nested(a, b)
    assert a["view_exposures"] == [3, 3, 3, 3]


def test_device_run_checkpoint_has_the_cpu_rng_schema(tmp_path, device, original_tests):
    """A paper-v1 checkpoint carries a CPU generator payload on every backend.

    This is why the lane needs no ``torch_xpu`` RNG key and why a PVC
    checkpoint stays readable by the unchanged CUDA code (audit C.1).
    """
    recipe = original_tests.tiny_recipe()
    out = tmp_path / "run"
    workflow_pvc.run(original_tests.TinyAcquisition(), recipe, out, device=device)
    state = original_tests.load(out / "checkpoint_final.pt")
    generator_state = state["generator_state"]
    assert torch.is_tensor(generator_state) and generator_state.dtype == torch.uint8
    assert generator_state.device.type == "cpu"
    assert not any(k.startswith("torch_xpu") or k.startswith("torch_cuda") for k in state)
    # every persisted tensor is on CPU, so the checkpoint is backend-neutral
    for key in ("fields", "iso_points", "iso_normals"):
        value = state.get(key)
        if torch.is_tensor(value):
            assert value.device.type == "cpu", key


def test_production_lane_never_creates_a_device_generator(tmp_path, device, original_tests, monkeypatch):
    """Regression guard for the XPU JIT hang (dispatch rule 5).

    ``torch.Generator(device="xpu")`` never finishes compiling on PVC. The
    paper-v1 lane must only ever build CPU generators; this fails loudly if
    that ever changes.
    """
    real_generator = torch.Generator
    created = []

    def spy(*args, **kwargs):
        dev = kwargs.get("device", args[0] if args else "cpu")
        created.append(str(dev))
        if torch.device(dev).type != "cpu":
            raise AssertionError(
                f"paper-v1 lane created a non-CPU torch.Generator(device={dev!r}); "
                "this hangs the Intel JIT on PVC")
        return real_generator(*args, **kwargs)

    monkeypatch.setattr(torch, "Generator", spy)
    workflow_pvc.run(original_tests.TinyAcquisition(), original_tests.tiny_recipe(),
                     tmp_path / "run", device=device)
    assert created, "expected the workflow to build at least one CPU generator"
    assert all(torch.device(d).type == "cpu" for d in created)
