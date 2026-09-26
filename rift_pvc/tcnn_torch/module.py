"""Shared base of the tinycudann-compatible shim modules.

Mirrors the observable contract of ``tinycudann.modules.Module`` (pinned
``bindings/torch/tinycudann/modules.py``): an ``nn.Module`` whose only
registered tensor is one flat ``params`` Parameter kept in float32 (TCNN keeps
``initial_params`` in fp32 and casts to ``param_precision()`` at forward), the
attributes ``n_input_dims``, ``n_output_dims``, ``seed``, ``dtype``,
``loss_scale`` and ``native_tcnn_module`` (``None`` here), inputs cast to
float32, and outputs truncated to ``n_output_dims``.

Precision (ledger D3): the shim computes in float32 by default.
``RIFT_PVC_TCNN_HALF=1`` selects the fp16 parity mode, in which parameters are
cast to float16 at forward and outputs are float16, as the CUDA build does with
``param_precision() == Fp16``. The mode is read once per module at construction.
"""
from __future__ import annotations

import os

import torch
from torch import nn

DEFAULT_SEED = 1337   # tinycudann.modules.Module(seed=1337)
HALF_ENV = "RIFT_PVC_TCNN_HALF"


def precision_mode() -> str:
    """``"fp16"`` when ``RIFT_PVC_TCNN_HALF=1``, else ``"fp32"``."""
    return "fp16" if os.environ.get(HALF_ENV, "").strip() == "1" else "fp32"


def compute_dtype() -> torch.dtype:
    return torch.float16 if precision_mode() == "fp16" else torch.float32


def resolve_dtype(dtype) -> torch.dtype:
    """``tcnn.Encoding(dtype=...)``: ``None`` means the build's preferred precision."""
    if dtype is None:
        return compute_dtype()
    if dtype in (torch.float32, torch.float16):
        return dtype
    raise ValueError(f"Encoding only supports fp32 or fp16 precision, but got {dtype}")


class _ScaleGrad(torch.autograd.Function):
    """Identity in the forward pass; scales the gradient in the backward pass."""

    @staticmethod
    def forward(ctx, x, factor):
        ctx.factor = factor
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.factor, None


def scale_grad(x: torch.Tensor, factor: float) -> torch.Tensor:
    """tiny-cuda-nn loss scaling (``modules.py`` ``_module_function``): in the fp16
    build dL/dy is multiplied by ``loss_scale`` before the half-precision backward
    and the parameter/input gradients are divided by it afterwards, so small
    gradients do not underflow in fp16. Applied only in the shim's fp16 mode."""
    return x if factor == 1.0 else _ScaleGrad.apply(x, factor)


def default_loss_scale(dtype: torch.dtype) -> float:
    """``common.h`` ``default_loss_scale``: 128 for fp16, 1 for fp32 (informational here)."""
    return 128.0 if dtype == torch.float16 else 1.0


class ShimModule(nn.Module):
    """Base class: one flat float32 ``params`` Parameter, TCNN attribute names."""

    def __init__(self, n_input_dims: int, n_output_dims: int, seed: int, dtype: torch.dtype):
        super().__init__()
        self.n_input_dims = int(n_input_dims)
        self.n_output_dims = int(n_output_dims)
        self.seed = int(seed)
        self.dtype = dtype
        self.loss_scale = default_loss_scale(dtype)
        self.native_tcnn_module = None

    def _generator(self) -> torch.Generator:
        # TCNN seeds a pcg32 with ``seed``; the stream itself is not reproducible
        # (ledger D2, "initial values differ in value but not in distribution").
        return torch.Generator().manual_seed(self.seed)

    def _set_params(self, initial: torch.Tensor) -> None:
        self.params = nn.Parameter(initial.to(torch.float32).contiguous(), requires_grad=True)

    def _check_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.n_input_dims:
            raise ValueError(f"{type(self).__name__}: expected input [N, {self.n_input_dims}], got {tuple(x.shape)}")
        return x.to(torch.float32)   # bindings: x_padded.to(torch.float)

    def extra_repr(self) -> str:
        return (f"n_input_dims={self.n_input_dims}, n_output_dims={self.n_output_dims}, "
                f"seed={self.seed}, dtype={self.dtype}, hyperparams={self.hyperparams()}")

    def hyperparams(self) -> dict:  # pragma: no cover - overridden
        return {}
