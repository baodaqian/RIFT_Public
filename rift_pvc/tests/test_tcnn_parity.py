"""Gate 2: replay of the real tiny-cuda-nn modules dumped on an H100
(``scripts_pvc/parity_tcnn_dump.py``) through the torch shim.

The dump (NPZ + JSON sidecar) is located by ``RIFT_PVC_TCNN_PARITY_NPZ`` or
the newest ``tcnn_parity_h100_*.npz`` under
``/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageE/parity``; without it
the tests skip. Params are copied through the layout map (identical flat
order), so ``shim.params.copy_(tcnn.params)`` gives the same function.
Tolerances: against the fp32-precision twins the formulas must agree to 1e-4
(grid; a device-``exp2f`` ulp on the level scale) and 1e-6 (SH). Against the
fp16 production modules the differences are tiny-cuda-nn's own fp16
arithmetic: at initialization scale (features ~1e-4, fp16 subnormals) 1.6e-3
for the grid and 2.4e-3 for ``xyz_net``, ~5e-4 on the amplified records
(grid x64, fp16 normal range) of dump 2154357. Forward is asserted at 5e-3
(init scale) / 2e-3 (amplified); gradients are asserted on the amplified
records (1e-2 parameters, 5e-2 inputs; input gradients are never needed in
production) and only reported at init scale, where the fp16 reference
gradients themselves are subnormal. Runs on CPU, XPU or CUDA; every number
is printed for the record.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rift_pvc import accelerator  # noqa: E402
from rift_pvc import tcnn_torch as tcnn  # noqa: E402

PARITY_DIR = "/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageE/parity"
BASE_MODULES = ("encode_xyz", "encode_angle", "xyz_net", "alpha_net", "rd_net")


def locate_dump():
    path = os.environ.get("RIFT_PVC_TCNN_PARITY_NPZ")
    if path:
        return Path(path)
    found = sorted(f for f in glob.glob(f"{PARITY_DIR}/tcnn_parity_h100_*.npz") if not f.endswith(".state.npz"))
    return Path(found[-1]) if found else None


DUMP = locate_dump()
pytestmark = pytest.mark.skipif(DUMP is None or not DUMP.is_file(), reason="no H100 tiny-cuda-nn dump available")
META = json.loads(Path(str(DUMP)[:-4] + ".json").read_text()) if DUMP is not None and DUMP.is_file() else {"modules": {}}
MODULES = tuple(META["modules"]) or BASE_MODULES
REDUCTION = META.get("reduction", "sum")


def reduce(out):
    return out.float().mean() if REDUCTION == "mean" else out.float().sum()


@pytest.fixture(scope="module")
def dump():
    data = np.load(DUMP)
    meta = json.loads(Path(str(DUMP)[:-4] + ".json").read_text())
    return data, meta


@pytest.fixture(scope="module")
def device():
    return accelerator.device() if accelerator.is_available() else torch.device("cpu")


def rel_err(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def build(meta, name, dtype=None):
    spec = meta["modules"][name]
    if spec["kind"] == "encoding":
        return tcnn.Encoding(spec["n_input_dims"], spec["config"], dtype=dtype)
    return tcnn.Network(spec["n_input_dims"], spec["n_output_dims"], spec["config"])


def loaded(meta, data, name, device, dtype=None):
    module = build(meta, name, dtype=dtype).to(device)
    params = torch.from_numpy(data[f"{name}/params"])
    assert module.params.numel() == params.numel() == meta["modules"][name]["n_params"], name
    with torch.no_grad():
        module.params.copy_(params.to(device))
    return module


@pytest.mark.parametrize("name", MODULES)
def test_layout_reproduces_tcnn_param_count_and_output_dims(dump, name):
    data, meta = dump
    module = build(meta, name)
    spec = meta["modules"][name]
    assert module.params.numel() == spec["n_params"]
    assert module.n_output_dims == spec["n_output_dims"]
    if spec["kind"] == "network":
        assert module.padded_output_width == spec["padded_output_width"]
    print(f"{name}: n_params {spec['n_params']} param_precision {spec['param_precision']} "
          f"output_precision {spec['output_precision']} hyperparams {spec['hyperparams']}")


@pytest.mark.parametrize("name", [m for m in MODULES if m.startswith("encode_")])
def test_encoding_forward_and_gradients_against_fp32_precision_tcnn(dump, name, device, monkeypatch):
    """The dump also evaluates each encoding as tcnn.Encoding(dtype=float32): tight check."""
    data, meta = dump
    monkeypatch.delenv("RIFT_PVC_TCNN_HALF", raising=False)
    key = f"{name}/fp32"
    if f"{key}/output" not in data:
        pytest.skip("dump has no fp32-precision twin")
    module = loaded(meta, data, name, device, dtype=torch.float32)
    x = torch.from_numpy(data[f"{name}/input"]).to(device).requires_grad_(True)
    out = module(x)
    err = rel_err(out.detach().cpu().numpy(), data[f"{key}/output"])
    reduce(out).backward()
    gp = rel_err(module.params.grad.cpu().numpy(), data[f"{key}/grad_params"]) if module.params.numel() else 0.0
    gx = rel_err(x.grad.cpu().numpy(), data[f"{key}/grad_input"])
    print(f"{name} fp32 twin: forward rel {err:.3e} grad_params rel {gp:.3e} grad_input rel {gx:.3e}")
    tolerance = 1e-6 if name.startswith("encode_angle") else 1e-4
    assert err < tolerance and gp < 1e-2 and gx < 1e-2


@pytest.mark.parametrize("name", MODULES)
@pytest.mark.parametrize("half", (False, True))
def test_forward_and_gradients_against_the_production_fp16_modules(dump, name, device, monkeypatch, half):
    data, meta = dump
    if half:
        monkeypatch.setenv("RIFT_PVC_TCNN_HALF", "1")
    else:
        monkeypatch.delenv("RIFT_PVC_TCNN_HALF", raising=False)
    if half and device.type == "cpu" and meta["modules"][name]["kind"] == "network":
        pytest.skip("fp16 matmul parity is a device check")
    module = loaded(meta, data, name, device)
    x = torch.from_numpy(data[f"{name}/input"]).to(device).requires_grad_(True)
    out = module(x)
    assert out.shape == tuple(data[f"{name}/output"].shape)
    err = rel_err(out.detach().float().cpu().numpy(), data[f"{name}/output"])
    reduce(out).backward()
    reference = data[f"{name}/grad_params"] if module.params.numel() else None
    if reference is not None and not np.isfinite(reference).all():
        pytest.skip(f"{name}: the tiny-cuda-nn reference gradient overflowed in its loss-scaled fp16 backward "
                    f"({int((~np.isfinite(reference)).sum())} non-finite entries; re-dump with a mean reduction)")
    gp = rel_err(module.params.grad.cpu().numpy(), reference) if reference is not None else 0.0
    gx = rel_err(x.grad.cpu().numpy(), data[f"{name}/grad_input"]) if f"{name}/grad_input" in data else float("nan")
    print(f"{name} [{'fp16' if half else 'fp32'} shim vs fp16 tcnn on {device}]: forward rel {err:.3e} "
          f"grad_params rel {gp:.3e} grad_input rel {gx:.3e}")
    amplified = name.endswith("@amp")
    # At initialization scale (features ~1e-4, mean-reduced loss) the fp16 reference gradients sit in fp16's
    # subnormal range (TCNN computes its input gradients in fp32; the shim's fp16 mode does not), so gradient
    # parity is asserted on the amplified records and reported at init scale.
    assert err < (2e-3 if amplified else 5e-3), f"{name}: forward relative error {err:.3e}"
    if amplified:
        assert gp < 1e-2, f"{name}: parameter gradient relative error {gp:.3e}"
        if not np.isnan(gx):
            assert gx < 5e-2, f"{name}: input gradient relative error {gx:.3e}"


@pytest.mark.parametrize("variant", ("model", "model@amp"))
def test_end_to_end_radarfield_matches_the_h100_forward(dump, device, monkeypatch, variant):
    """Whole RadarField (eval mode, BN statistics from the dump) through the shim."""
    data, meta = dump
    if f"{variant}/alpha" not in data:
        pytest.skip(f"dump has no {variant} record")
    monkeypatch.delenv("RIFT_PVC_TCNN_HALF", raising=False)
    monkeypatch.setenv("RIFT_PVC_TCNN_SHIM", "1")
    from rift_pvc.radar_fields_upstream import install_shim, original_module
    install_shim()
    kwargs = meta["model"]["radarfield_kwargs"]
    model = original_module("radarfields.nn.models").RadarField(**kwargs).to(device)
    state = {k: torch.from_numpy(v) for k, v in np.load(str(DUMP)[:-4] + ".state.npz").items()}
    if variant.endswith("@amp"):
        state["encode_xyz.params"] = state["encode_xyz.params"] * meta["amp_factor"]
    model.load_state_dict(state, strict=True)
    model.eval()
    xyz = torch.from_numpy(data["model/xyz"]).to(device)
    angle = torch.from_numpy(data["model/angle"]).to(device)
    with torch.no_grad():
        out = model(xyz, angle, sin_epoch=meta["model"]["sin_epoch"])
    ea = rel_err(out["alpha"].float().cpu().numpy(), data[f"{variant}/alpha"])
    er = rel_err(out["rd"].float().cpu().numpy(), data[f"{variant}/rd"])
    print(f"RadarField end-to-end ({variant}) on {device}: alpha rel {ea:.3e} rd rel {er:.3e}")
    assert ea < 2e-3 and er < 2e-3
