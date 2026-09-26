"""CPU-generator sampling for the Sugavanam--Ertin Stage-2 lane on XPU.

``train_sugavanam_ertin_stage2.py`` builds ``torch.Generator(device=device)``
(L1325) and then draws with ``torch.randint(..., generator=g, device=device)``
and ``sample_roi(n, extent, g, device)``. On PVC a device generator must never
be created -- the Intel JIT never finishes compiling it (dispatch rule 5) --
and a CPU generator cannot be passed to a call that allocates on ``xpu``:
PyTorch requires the generator and the target device to agree.

So on XPU the draws happen on the CPU generator and the samples are moved to
the device. This module provides the two pieces that makes that possible
without copying the 510-line ``_run``:

* :func:`sample_roi` -- a drop-in twin of
  ``rift.sugavanam_ertin_validzero.sample_roi``.
* :class:`CpuGeneratorTorch` -- a narrow proxy for the ``torch`` module that
  constructs XPU-requested generators on CPU and adapts ``randint``, ``rand``,
  ``randn`` and ``randperm`` draws; every other attribute is the real
  ``torch``. The Stage-2 PVC entry point binds it into that module's
  namespace, the same in-process rebinding ``train_pvc.py`` uses.

**Recorded consequence** (dispatch rule 4): the random stream is the CPU
stream, not a device stream, so an XPU Stage-2 run is *state-compatible but
not trajectory-identical* to a CUDA run. This is the same branch the original
already takes for its CPU lane -- ``_capture_rng(include_cuda=False)`` -- so
the checkpoint payload stays inside the closed schema
``{python, numpy, torch_cpu, torch_cuda}`` that
``_validate_rng_state`` enforces, and a PVC checkpoint remains readable by the
unchanged CUDA code. No ``torch_xpu`` key is added, deliberately.
"""
from __future__ import annotations

import torch

__all__ = ["sample_roi", "CpuGeneratorTorch", "draws_on_cpu"]


def draws_on_cpu(generator, device) -> bool:
    """True when the draw must happen on CPU and the result be moved."""
    if generator is None:
        return False
    generator_device = getattr(generator, "device", torch.device("cpu"))
    device = torch.get_default_device() if device is None else device
    return torch.device(generator_device).type == "cpu" and torch.device(device).type != "cpu"


def sample_roi(n, extent, generator, device, dtype=torch.float32):
    """Twin of ``rift.sugavanam_ertin_validzero.sample_roi``.

    Identical formula and identical validation; only the draw location changes
    when a CPU generator is used with an accelerator device.
    """
    if n <= 0:
        raise ValueError("sample count must be positive")
    if draws_on_cpu(generator, device):
        values = torch.rand(n, 3, generator=generator, dtype=dtype)
        return ((2.0 * values - 1.0) * extent).to(device)
    return (2.0 * torch.rand(n, 3, generator=generator, device=device, dtype=dtype) - 1.0) * extent


class CpuGeneratorTorch:
    """The ``torch`` module with XPU generators and their draws moved to CPU.

    Every other attribute -- ``torch.Tensor``, ``torch.optim``, ``torch.load``,
    ``torch.cuda`` -- is looked up on the real module, so the proxy changes
    nothing else about the code that uses it.
    """

    OVERRIDDEN = ("Generator", "randint", "rand", "randn", "randperm")

    def __init__(self, module=torch):
        object.__setattr__(self, "_module", module)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_module"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_module"), name, value)

    def Generator(self, device="cpu"):
        module = object.__getattribute__(self, "_module")
        return module.Generator(device="cpu" if torch.device(device).type == "xpu" else device)

    def _draw(self, name, *args, **kwargs):
        module = object.__getattribute__(self, "_module")
        generator = kwargs.get("generator")
        device = kwargs.get("device")
        device = module.get_default_device() if device is None else device
        if draws_on_cpu(generator, device):
            kwargs = dict(kwargs)
            kwargs["device"] = "cpu"
            return getattr(module, name)(*args, **kwargs).to(device)
        return getattr(module, name)(*args, **kwargs)

    def randint(self, *args, **kwargs):
        return self._draw("randint", *args, **kwargs)

    def rand(self, *args, **kwargs):
        return self._draw("rand", *args, **kwargs)

    def randn(self, *args, **kwargs):
        return self._draw("randn", *args, **kwargs)

    def randperm(self, *args, **kwargs):
        return self._draw("randperm", *args, **kwargs)
