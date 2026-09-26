"""The faithful twin of GeRaF's ``torch.amp.autocast("cuda", float16)`` regions.

``torch.amp.autocast("cuda", dtype=torch.float16)`` is *disabled* on a host
without CUDA — torch prints "CUDA is not available ... Disabling autocast" and
every op keeps its input dtype. ``torch.autocast("cpu", dtype=torch.float16)``,
by contrast, really does run the region in float16, which would change the CPU
numerics that the GeRaF tests and the CPU reference runs pin.

So the twin enables float16 autocast on the active accelerator and leaves it
disabled on the CPU backend, which reproduces the original's behaviour on both
kinds of host. It is expressed through ``rift_pvc.accelerator.autocast`` only;
it is not a second shim.
"""
from __future__ import annotations

import torch

from rift_pvc import accelerator


def autocast_fp16():
    """float16 autocast on CUDA/XPU; a disabled context on the CPU backend."""
    return accelerator.autocast(torch.float16, enabled=accelerator.is_available())
