"""GeRaF PVC device gates: run the adapted code on a PVC card when one is present.

Skipped without an allocated XPU. The CPU leg of each parameterized test is the
reference the device leg is compared against; the CPU numbers themselves are
pinned by ``rift_pvc/tests/test_geraf_pvc.py`` against the unchanged CUDA
modules.
"""
from __future__ import annotations

import copy
import json
import signal as signal_module
import time
import warnings

import numpy as np
import pytest
import torch

from rift.geraf_source_ops import NativeAcquisition, bilinear_ray_sampler
from rift_pvc import geraf_source as source
from rift_pvc import geraf_source_training as runtime
from rift_pvc.geraf_autocast import autocast_fp16
from tests.test_geraf_source import SMALL, TinyData, acquisition, compare_nested

FALLBACK = "fallback from XPU to CPU"


@pytest.fixture(params=["cpu", "xpu"])
def device(request, monkeypatch):
    if request.param == "xpu" and not torch.xpu.is_available():
        pytest.skip("requires an allocated PVC card")
    monkeypatch.setenv("RIFT_ACCELERATOR", request.param)
    return torch.device(request.param)


@pytest.fixture
def xpu(monkeypatch):
    if not torch.xpu.is_available():
        pytest.skip("requires an allocated PVC card")
    monkeypatch.setenv("RIFT_ACCELERATOR", "xpu")
    return torch.device("xpu")


def on_device(op: NativeAcquisition, device) -> NativeAcquisition:
    return NativeAcquisition(op.tx.to(device), op.rx.to(device), op.frequencies.to(device),
                             op.reference_path.to(device), phase_sign=op.phase_sign,
                             point_chunk=op.point_chunk, pair_chunk=op.pair_chunk,
                             uniform_rift=op.uniform_rift)


class DeviceTinyData(TinyData):
    """``TinyData`` with its acquisition and response placed on the device."""

    def __init__(self, device):
        super().__init__()
        self.device = device

    def acquisition(self, role, view, head, recipe, device):
        self.key(role, view, head)
        return on_device(acquisition(), self.device)

    def response(self, role, view, head, device):
        self.key(role, view, head)
        self.reads.append((role, view))
        return torch.ones(2, 8, dtype=torch.complex128, device=self.device)


# --------------------------------------------------------------------------
# Kernels
# --------------------------------------------------------------------------

def test_ray_interpolation_bounds_and_denominator_on_device(device):
    """Device twin of tests/test_geraf_source.py::
    test_ray_interpolation_cuda_bounds_and_denominator (the original keeps its name)."""
    z = torch.tensor([[0., 1e-7, 1.]], dtype=torch.float64, device=device, requires_grad=True)
    q = torch.tensor([[-1., 0., 5e-8, 1., 2.]], dtype=torch.float64, device=device)
    n = torch.arange(9., dtype=torch.float64, device=device).reshape(1, 3, 3).requires_grad_()
    s = torch.tensor([[[1., 3., 7.]]], dtype=torch.float64, device=device, requires_grad=True)
    normals, sigmas, inside = bilinear_ray_sampler(n, s, z, q)
    torch.testing.assert_close(sigmas.cpu(), torch.tensor([[[0., 1., 1.1, 7., 0.]]], dtype=torch.float64))
    assert inside.cpu().tolist() == [[False, True, True, True, False]]
    (normals.sum() + sigmas.sum()).backward()
    assert z.grad is None                       # geometry stays detached, as upstream
    assert n.grad is not None and s.grad is not None


def test_native_trace_and_matched_filter_agree_with_cpu(xpu):
    """fp64/complex128 tracer and matched filter on the card."""
    op = acquisition()
    op.reference_path[:] = torch.tensor([3.9, 4.1])
    xyz = torch.tensor([[0., 0., 0.], [.01, .02, 0.], [-.02, .01, 0.]], dtype=torch.float64)
    n = torch.tensor([[1., 0., 0.], [1., .1, 0.], [1., 0., .1]], dtype=torch.float64)
    sigmas = torch.tensor([[.3, .4, .5], [.6, .7, .8]], dtype=torch.float64)
    inside = torch.tensor([True, False, True])
    expected = op.trace(n.clone().requires_grad_(), sigmas.clone().requires_grad_(), xyz, op.tx, op.rx, inside)
    gpu = on_device(op, xpu)
    nd, sd = n.to(xpu).requires_grad_(), sigmas.to(xpu).requires_grad_()
    actual = gpu.trace(nd, sd, xyz.to(xpu), gpu.tx, gpu.rx, inside.to(xpu))
    assert actual.dtype == torch.complex128 and actual.device.type == "xpu"
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-10, atol=1e-13)
    torch.autograd.grad(actual.abs().square().sum(), (nd, sd))
    response = torch.complex(torch.arange(16.).reshape(2, 8).double(), torch.ones(2, 8).double())
    torch.testing.assert_close(gpu.matched_filter(response.to(xpu), xyz.to(xpu)).cpu(),
                               op.matched_filter(response, xyz), rtol=1e-10, atol=1e-12)


def test_lattice_and_measured_values_agree_with_cpu(xpu):
    recipe = source.recipe_from_config(SMALL, .15)
    indices = np.arange(recipe['mf_grid'] ** 3)
    host = runtime.lattice_points(indices, recipe['mf_grid'], recipe['extent_m'], 'cpu')
    card = runtime.lattice_points(indices, recipe['mf_grid'], recipe['extent_m'], xpu)
    torch.testing.assert_close(card.cpu(), host, rtol=0, atol=0)
    op, response = acquisition(), torch.ones(2, 8, dtype=torch.complex128)
    expected = runtime.measured_values(op, response, host, recipe['trans_power'])
    actual = runtime.measured_values(on_device(op, xpu), response.to(xpu), card, recipe['trans_power'])
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-13)


# --------------------------------------------------------------------------
# The fp16 autocast path: the one numerical risk of this port
# --------------------------------------------------------------------------

def test_autocast_is_float16_on_the_card_and_inert_on_cpu(device):
    a = torch.randn(64, 64, device=device)
    with autocast_fp16():
        dtype = (a @ a).dtype
    assert dtype == (torch.float16 if device.type == "xpu" else torch.float32)


def test_device_fp16_loss_tracks_the_cpu_fp32_reference(xpu, capsys):
    """The dispatch's explicit ask: no CUDA GeRaF run exists to compare against,
    so validate the fp16 autocast path against a small CPU fp32 run and report
    the numerical difference."""
    recipe = source.recipe_from_config(SMALL, .15)
    torch.manual_seed(recipe['seed'])
    reference = source.build_model(recipe, 'cpu')
    state = copy.deepcopy(reference.state_dict())
    card = source.build_model(recipe, 'cpu')
    card.load_state_dict(state, strict=True)
    card = card.to(xpu)

    cube = np.ones((SMALL['mf_grid'],) * 3, np.float32)
    op = acquisition()
    with source.fixed_numpy_seed(9):
        host_frame = source.sample_frame(op, recipe, 'a', cube, cube)
    with source.fixed_numpy_seed(9):
        card_frame = source.sample_frame(on_device(op, xpu), recipe, 'a', cube, cube)

    reference.radar_cfg = {'native': op}
    card.radar_cfg = {'native': on_device(op, xpu)}
    host = {k: float(v) for k, v in reference.loss(host_frame).items()}
    device_losses = {k: float(v) for k, v in card.loss(card_frame).items()}
    assert set(host) == set(device_losses)
    report = {}
    for key, expected in host.items():
        got = device_losses[key]
        assert np.isfinite(got)
        report[key] = dict(cpu_fp32=expected, xpu_fp16=got,
                           relative=abs(got - expected) / max(abs(expected), 1e-30))
    with capsys.disabled():
        print("\nGeRaF fp16-on-XPU vs fp32-on-CPU: " + json.dumps(report, indent=2))
    # fp16 autocast on the SDF/feature network. Measured on ac025 (job 2153307):
    # mf_loss 1.7e-7, grad_loss 2.3e-4 relative. The bound is ~40x the worst of
    # those: loose enough for node-to-node variation, tight enough that a broken
    # autocast region (fp16 leaking into the tracer, or autocast silently off)
    # fails here.
    assert report['mf_loss']['relative'] < 1e-2
    assert report['grad_loss']['relative'] < 1e-2


# --------------------------------------------------------------------------
# Full lifecycle on the card
# --------------------------------------------------------------------------

def test_source_lifecycle_trains_checkpoints_and_resumes_on_the_card(xpu, tmp_path):
    data = DeviceTinyData(xpu)
    started = time.monotonic()
    complete = runtime.train(data=data, output_dir=tmp_path / 'full', config=SMALL, device=xpu)
    assert complete['status'] == 'complete'
    seconds_per_step = (time.monotonic() - started) / SMALL['steps']

    interrupted_dir = tmp_path / 'interrupted'
    original_step = torch.optim.AdamW.step
    calls = [0]

    def stop_after_first(optimizer, *args, **kwargs):
        result = original_step(optimizer, *args, **kwargs)
        calls[0] += 1
        if calls[0] == 1:
            signal_module.getsignal(signal_module.SIGTERM)(signal_module.SIGTERM, None)
        return result

    with pytest.MonkeyPatch.context() as m:
        m.setattr(torch.optim.AdamW, 'step', stop_after_first)
        assert runtime.train(data=DeviceTinyData(xpu), output_dir=interrupted_dir,
                             config=SMALL, device=xpu)['status'] == 'interrupted'
    saved = torch.load(interrupted_dir / 'checkpoint_latest.pth.tar', map_location='cpu', weights_only=False)
    assert saved['rng_cuda'] == [] and len(saved['rng_xpu']) == torch.xpu.device_count()
    resumed = runtime.train(data=DeviceTinyData(xpu), output_dir=interrupted_dir,
                            config=SMALL, device=xpu, resume='auto')
    assert resumed['status'] == 'complete'

    a = torch.load(tmp_path / 'full' / 'checkpoint_latest.pth.tar', map_location='cpu', weights_only=False)
    b = torch.load(interrupted_dir / 'checkpoint_latest.pth.tar', map_location='cpu', weights_only=False)
    compare_nested(a['models'], b['models'])
    compare_nested(a['optimizer'], b['optimizer'])
    assert a['validation_history'] == b['validation_history']
    assert a['complete'] and a['validation_history'][-1]['masked'] is False
    print(f"\nGeRaF synthetic lifecycle on XPU: {seconds_per_step:.3f} s/step "
          f"({SMALL['steps']} steps incl. validation)")


def test_checkpoint_written_on_the_card_loads_back_with_xpu_map_location(xpu, tmp_path):
    data = DeviceTinyData(xpu)
    runtime.train(data=data, output_dir=tmp_path / 'run', config=SMALL, device=xpu)
    payload = torch.load(tmp_path / 'run' / 'checkpoint_best.pth.tar',
                         map_location='xpu:0', weights_only=False)
    models, row = runtime.load_selected_models(payload, data, xpu)
    assert row['masked'] is False
    model = next(iter(models.values()))
    assert {p.device.type for p in model.parameters()} == {'xpu'}
    # weight_norm's `weight` is a plain attribute recomputed by a forward hook,
    # so it only reaches the device once the model runs.
    assert model.sdf_network.lin0.weight_v.device.type == 'xpu'


def test_no_aten_operator_falls_back_to_the_cpu(xpu, tmp_path):
    """PYTORCH_DEBUG_XPU_FALLBACK=1 turns a silent fallback into a warning."""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        runtime.train(data=DeviceTinyData(xpu), output_dir=tmp_path / 'run', config=SMALL, device=xpu)
    offenders = [str(w.message) for w in captured if FALLBACK in str(w.message)]
    assert offenders == []
