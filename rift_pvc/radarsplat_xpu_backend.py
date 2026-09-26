"""PVC backend for the released RadarSplat renderer: ``fork_torch_mirror_xpu_v1``.

Package F, decisions D1/D4/D5 (``docs/RADARSPLAT_PVC_ADAPTATION.md``).
``load_xpu_reference`` is the twin of ``rift.radarsplat_release.load_cuda_reference``:
it verifies the pinned fork inventory, imports the fork's ``gsplat.rendering``
without ever consulting a CUDA toolkit, binds the five CUDA ops the radar
branch reaches to their torch mirrors (``rift_pvc.gsplat_torch_ops``) and
returns ``(rendering, fused_ssim)`` where ``fused_ssim`` is the torch twin
(``rift_pvc.fused_ssim_torch``). The fork's ``_radar_rasterization`` and
``_rasterize_to_radar_pixels`` then run as written, with one exception:

F-dev1: five literal CUDA constructors sit on the production path of the
fork's Python: ``_radar_rasterization`` (``torch.ones(...).to('cuda')`` for
the homogeneous means, ``torch.zeros((C, 4, 4)).to('cuda')`` for the identity
view matrices), the filters ``spectral_leakage`` and
``azimuth_antenna_gain_projection`` (``kernel = ....cuda()``) and the
preprocessing ``boreas/data_processing/play_radar_signal.py::FFT``
(``torch.tensor(...).to("cuda")``). The loader re-executes each function's own
source with exactly those literals replaced by the input tensor's device
(``means.device``, ``raw_image.device``) or, for the preprocessing FFT, the
accelerator device; every replacement count is asserted so a changed pinned
source is noticed. Nothing else in those functions changes.

Identity (D5): every PVC checkpoint directory carries ``backend.json``
(``radarsplat_backend``, ``ssim``, torch/shim facts, parity job). The release
checkpoint's ``identity`` dict is compared for equality on resume, so the
backend is deliberately not added to it. A checkpoint directory without the
sidecar is a CUDA run; continuing one on the mirror is refused unless
``RIFT_PVC_RADARSPLAT_ALLOW_CUDA_RESUME=1``.
"""
from __future__ import annotations

import importlib
import inspect
import json
import os
import sys
import types
from pathlib import Path
from typing import Mapping, Optional

import torch

from rift.radarsplat_release import COMMIT, REFERENCE_ROOT, reference_contract, verify_reference
from rift_pvc import accelerator
from rift_pvc import fused_ssim_torch
from rift_pvc import gsplat_torch_ops as ops

BACKEND_IDENTITY = "fork_torch_mirror_xpu_v1"
SSIM_IDENTITY = fused_ssim_torch.IDENTITY
SIDECAR_NAME = "backend.json"
SIDECAR_SCHEMA = "rift_pvc_radarsplat_backend_v1"
BACKEND_MODULE = "gsplat.cuda._backend"
CUDA_LITERAL = ".to('cuda')"
DEVICE_NEUTRAL = ".to(means.device)"
EXPECTED_CUDA_LITERALS = 2

RENDERING_REBINDS = {
    "fully_fused_projection": ops.fully_fused_projection_xpu,
    "isect_tiles": ops.isect_tiles_xpu,
    "isect_offset_encode": ops.isect_offset_encode_xpu,
    "spherical_harmonics": ops.spherical_harmonics_xpu,
}
WRAPPER_REBINDS = {
    **RENDERING_REBINDS,
    "rasterize_to_indices_in_range_radargs": ops.rasterize_to_indices_in_range_radargs_xpu,
}
# Gate 1 outcome (RIFT_PVC_Adaptation.md 8.F, docs/RADARSPLAT_PVC_ADAPTATION.md); updated when a reference is replayed.
PARITY_JOBS = ("H100 references: 2153858 (cuDNN TF32 on) and 2154965 (NVIDIA_TF32_OVERRIDE=0, self-checked, GPU 41496d0c); "
               "PVC replays 2154368 (TF32-on) and 2155075/2155076 (TF32-off, three fixtures)")
PARITY_STATUS = ("gate 1 passed on XPU against fp32 CUDA arithmetic: projection radii identical, conics 1.5e-7, tile keys/offsets/SH "
                 "identical, radar index pairs and order identical for all 18 calls, products <= 2.9e-5 abs, fused_ssim 4.7e-10; "
                 "end to end: images <= 3.1e-5 abs, loss <= 3.4e-7 rel, gradients L2 <= 7.0e-4 (initial isotropic scene, one "
                 "near-zero-gradient Gaussian at 5.6e-3 of the max norm) and <= 1.2e-4 L2 / 3.8e-4 max norm on the perturbed "
                 "anisotropic scene; the production TF32-on reference differs from both by 2.3e-4 abs in the filtered images and "
                 "5.5e-5 rel in the loss (cuDNN TF32 convolutions)")
DEVIATIONS = [
    "F-dev1: five literal CUDA constructors in the fork's Python (_radar_rasterization x2, spectral_leakage, "
    "azimuth_antenna_gain_projection, play_radar_signal.FFT) are re-executed from their own source with the "
    "literal replaced by the input tensor's device (or the accelerator device for the preprocessing FFT)",
    "projection outputs of culled Gaussians and masked SH colours are finite/zero instead of uninitialised",
    "IEEE arithmetic instead of --use_fast_math/__expf; candidates within float rounding of the 1/255 "
    "cutoff may differ (gate 1 reports the count)",
    "isect_tiles is a vectorised mirror with the kernel's key encoding and a stable sort (the fork's own "
    "Python _isect_tiles is a per-Gaussian loop and is not used)",
    "spherical_harmonics evaluates at most degree 4 like the kernel (the fork's torch _spherical_harmonics "
    "leaves bases 25-35 uninitialised at degree 5 and is not used)",
    "fused_ssim is the torch twin fused_ssim_torch_v1 (separable 11-tap window, img2 detached)",
]


class _GuardedC:
    """Stands in for gsplat's compiled extension: any attribute access is a CUDA op reached on PVC."""

    def __getattr__(self, name):
        raise RuntimeError(
            f"gsplat CUDA op {name!r} was reached on the PVC backend ({BACKEND_IDENTITY}); "
            "every op the radar branch needs must be bound to its torch mirror")

    def __bool__(self):
        return True

    def __repr__(self):
        return "<rift_pvc guard: gsplat CUDA extension is not available on PVC>"


def install_backend_stub() -> types.ModuleType:
    """Seed ``gsplat.cuda._backend`` before the first gsplat import so no JIT build is attempted."""
    existing = sys.modules.get(BACKEND_MODULE)
    if existing is not None:
        if isinstance(getattr(existing, "_C", None), _GuardedC):
            return existing
        raise RuntimeError("gsplat's CUDA backend module was already imported in this process; "
                           "use a fresh process for the PVC RadarSplat backend")
    stub = types.ModuleType(BACKEND_MODULE)
    stub._C = _GuardedC()
    stub.__all__ = ["_C"]
    stub.__file__ = "<rift_pvc.radarsplat_xpu_backend stub: no CUDA toolkit is consulted>"
    sys.modules[BACKEND_MODULE] = stub
    return stub


# function name -> {literal: (replacement, expected count)}
RENDERING_DEVICE_LITERALS = {
    "_radar_rasterization": {".to('cuda')": (".to(means.device)", 2)},
    "spectral_leakage": {".cuda()": (".to(raw_image.device)", 1)},
    "azimuth_antenna_gain_projection": {".cuda()": (".to(raw_image.device)", 1)},
}
PREPROCESSING_SOURCE = "boreas/data_processing/play_radar_signal.py"
PREPROCESSING_FUNCTIONS = ("FFT", "multipath_modeling")
PREPROCESSING_DEVICE_NAME = "_RIFT_PVC_DEVICE"
PREPROCESSING_DEVICE_LITERALS = {'.to("cuda")': (f".to({PREPROCESSING_DEVICE_NAME})", 1)}
DEVICE_NEUTRAL_MARK = "__rift_pvc_device_neutral__"


def _replace_literals(source: str, replacements: Mapping[str, tuple], *, label: str) -> str:
    for literal, (replacement, expected) in replacements.items():
        count = source.count(literal)
        if count != expected:
            raise RuntimeError(
                f"expected exactly {expected} literal {literal} in the fork's {label}, found {count}; "
                "the pinned source changed and F-dev1 must be revisited")
        source = source.replace(literal, replacement)
    return source


def reexecute_device_neutral(module, name: str, replacements: Mapping[str, tuple]):
    """F-dev1: rebuild ``module.<name>`` from its own source with device-neutral constructors."""
    original = getattr(module, name)
    if getattr(original, DEVICE_NEUTRAL_MARK, False):
        return original
    source = _replace_literals(inspect.getsource(original), replacements, label=name)
    code = compile(source, module.__file__, "exec")
    namespace = module.__dict__  # the fork module's own globals: rebound ops, torch, F, math, Literal, ...
    exec(code, namespace)
    function = namespace[name]
    setattr(function, DEVICE_NEUTRAL_MARK, True)
    function.__rift_pvc_original__ = original
    return function


def device_neutral_radar_rasterization(rendering):
    """All of RENDERING_DEVICE_LITERALS; returns the rebuilt ``_radar_rasterization``."""
    for name, replacements in RENDERING_DEVICE_LITERALS.items():
        if not callable(getattr(rendering, name, None)):
            raise RuntimeError(f"the fork's gsplat.rendering no longer defines {name}; revisit F-dev1")
        reexecute_device_neutral(rendering, name, replacements)
    return rendering._radar_rasterization


def device_neutral_source_functions(relative, names, namespace, replacements, *, root=REFERENCE_ROOT):
    """``rift.radarsplat_release.source_functions`` on a source with F-dev1 literal replacements."""
    import ast
    root = verify_reference(root)
    path = root / relative
    text = _replace_literals(path.read_text(), replacements, label=relative)
    tree = ast.parse(text)
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in selected} != set(names):
        raise ValueError("Pinned source function inventory changed")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def preprocessing_functions(device, *, root=REFERENCE_ROOT):
    """The release's ``FFT``/``multipath_modeling`` with the FFT on ``device`` instead of CUDA."""
    import numpy as np
    from scipy.optimize import curve_fit
    namespace = dict(np=np, torch=torch, curve_fit=curve_fit)
    namespace[PREPROCESSING_DEVICE_NAME] = torch.device(device)
    return device_neutral_source_functions(PREPROCESSING_SOURCE, list(PREPROCESSING_FUNCTIONS), namespace,
                                           PREPROCESSING_DEVICE_LITERALS, root=root)


def bind_mirrors(rendering, wrapper) -> None:
    """Bind the five mirrors where the fork looks them up."""
    for name, mirror in RENDERING_REBINDS.items():
        if not callable(getattr(rendering, name, None)):
            raise RuntimeError(f"the fork's gsplat.rendering no longer defines {name}; revisit the PVC backend")
        setattr(rendering, name, mirror)
    for name, mirror in WRAPPER_REBINDS.items():
        if not callable(getattr(wrapper, name, None)):
            raise RuntimeError(f"the fork's gsplat.cuda._wrapper no longer defines {name}; revisit the PVC backend")
        setattr(wrapper, name, mirror)


def load_xpu_reference(root=REFERENCE_ROOT, *, device=None):
    """Twin of ``load_cuda_reference``: the fork's renderer on the torch mirrors, and the SSIM twin."""
    root = verify_reference(root)
    device = accelerator.device() if device is None else torch.device(device)
    if device.type == "cuda":
        raise ValueError("load_xpu_reference is the PVC backend; use rift.radarsplat_release.load_cuda_reference on CUDA")
    if device.type == "xpu" and not accelerator.is_available():
        raise RuntimeError("Released RadarSplat on PVC requires an allocated XPU device; no CPU/alternate-model fallback")
    for name in ("gsplat",):
        previous = sys.modules.get(name)
        if previous is not None and not Path(previous.__file__).resolve().is_relative_to(root):
            raise RuntimeError("Another gsplat was already imported; use a fresh process for RadarSplat")
    install_backend_stub()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    rendering = importlib.import_module("gsplat.rendering")
    wrapper = importlib.import_module("gsplat.cuda._wrapper")
    if not Path(rendering.__file__).resolve().is_relative_to(root):
        raise RuntimeError("gsplat.rendering was imported from outside the pinned RadarSplat source")
    backend = sys.modules.get(BACKEND_MODULE)
    if backend is None or not isinstance(getattr(backend, "_C", None), _GuardedC):
        raise RuntimeError("gsplat's compiled extension was reached; the PVC stub must be in place before import")
    bind_mirrors(rendering, wrapper)
    device_neutral_radar_rasterization(rendering)
    if device.type == "xpu":
        accelerator.set_device(device.index if device.index is not None else accelerator.current_device())
    return rendering, fused_ssim_torch.fused_ssim


def backend_record(*, parity_job: Optional[str] = None) -> dict:
    """Facts written to ``backend.json`` (D5) and copied into reports."""
    contract = reference_contract()
    return {
        "schema": SIDECAR_SCHEMA,
        "radarsplat_backend": BACKEND_IDENTITY,
        "ssim": SSIM_IDENTITY,
        "source_commit": COMMIT,
        "fused_ssim_commit": contract["fused_ssim_commit"],
        "mirrored_ops": sorted(WRAPPER_REBINDS),
        "deviations": list(DEVIATIONS),
        "torch": torch.__version__,
        "accelerator": accelerator.describe(),
        "parity_job": parity_job if parity_job is not None else os.environ.get("RIFT_PVC_RADARSPLAT_PARITY_JOB", PARITY_JOBS),
        "parity_status": os.environ.get("RIFT_PVC_RADARSPLAT_PARITY_STATUS", PARITY_STATUS),
    }


def read_sidecar(directory) -> Optional[Mapping[str, object]]:
    path = Path(directory) / SIDECAR_NAME
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _compatible(record: Mapping[str, object]) -> bool:
    return (record.get("schema") == SIDECAR_SCHEMA and record.get("radarsplat_backend") == BACKEND_IDENTITY
            and record.get("ssim") == SSIM_IDENTITY)


def write_sidecar(directory) -> Path:
    """Write ``backend.json`` once per checkpoint directory; refuse to overwrite another backend's."""
    directory = Path(directory)
    path = directory / SIDECAR_NAME
    existing = read_sidecar(directory)
    if existing is not None:
        if not _compatible(existing):
            raise RuntimeError(f"{path} names backend {existing.get('radarsplat_backend')!r}/{existing.get('ssim')!r}, "
                               f"not {BACKEND_IDENTITY!r}/{SSIM_IDENTITY!r}")
        return path
    directory.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(backend_record(), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
    return path


def check_resume_sidecar(directory) -> Optional[Mapping[str, object]]:
    """Before continuing a checkpoint directory: it must be a PVC run of this backend."""
    directory = Path(directory)
    latest = directory / "checkpoint_latest.pt"
    record = read_sidecar(directory)
    if not latest.exists():
        if record is not None and not _compatible(record):
            raise RuntimeError(f"{directory / SIDECAR_NAME} belongs to another backend")
        return record
    if record is None:
        if os.environ.get("RIFT_PVC_RADARSPLAT_ALLOW_CUDA_RESUME") == "1":
            print(f"radarsplat_xpu_backend: continuing {latest} without {SIDECAR_NAME} (a CUDA-renderer run) "
                  "because RIFT_PVC_RADARSPLAT_ALLOW_CUDA_RESUME=1; the result is a mixed-backend trajectory",
                  flush=True)
            return None
        raise RuntimeError(
            f"{latest} has no {SIDECAR_NAME}: it was produced by the released CUDA renderer and the PVC torch "
            f"mirror ({BACKEND_IDENTITY}) must not continue it (set RIFT_PVC_RADARSPLAT_ALLOW_CUDA_RESUME=1 "
            "to override deliberately)")
    if not _compatible(record):
        raise RuntimeError(f"{directory / SIDECAR_NAME} names backend {record.get('radarsplat_backend')!r}, "
                           f"not {BACKEND_IDENTITY!r}; refusing to continue")
    return record


__all__ = [
    "BACKEND_IDENTITY", "SSIM_IDENTITY", "SIDECAR_NAME", "SIDECAR_SCHEMA", "DEVIATIONS",
    "RENDERING_REBINDS", "WRAPPER_REBINDS", "RENDERING_DEVICE_LITERALS", "PREPROCESSING_DEVICE_LITERALS",
    "install_backend_stub", "reexecute_device_neutral", "device_neutral_radar_rasterization",
    "device_neutral_source_functions", "preprocessing_functions",
    "bind_mirrors", "load_xpu_reference", "backend_record", "read_sidecar", "write_sidecar",
    "check_resume_sidecar",
]
