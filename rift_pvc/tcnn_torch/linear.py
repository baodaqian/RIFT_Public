"""Bias-free linear layer with a selectable, order-deterministic weight-gradient reduction.

Why: on PVC, oneDNN's float32 GEMM splits the reduction dimension for the
weight gradient ``grad_out.T @ x`` (K = number of queries, ~1e6 per Radar
Fields step) and accumulates the partial sums in a nondeterministic order, so
two runs with identical seeds diverge from the last bits (measured in
``RIFT_PVC_Adaptation.md``, Package E). tiny-cuda-nn's own weight gradient
(``fc_multiply_split_k``, CUTLASS parallel split-K) reduces its partials in a
fixed order, so a fixed-order reduction is the faithful choice. The forward
GEMM (K = input width) and the input gradient (K = output width) have small K
and are deterministic.

Strategies (``RIFT_PVC_TCNN_WEIGHT_GRAD``):

* ``bmm:<block>``  partial GEMMs over row blocks of ``block`` queries
  (``torch.bmm``, K = block, parallel over blocks), then a fixed-order
  ``sum(0)``; rows are zero-padded to a multiple of ``block``;
* ``outer:<chunk>`` chunked outer products reduced with ``sum(0)`` (slow,
  reference path);
* ``fp64``           the plain GEMM in float64;
* ``gemm``           the plain float32 GEMM (torch autograd behaviour).
"""
from __future__ import annotations

import os

import torch

ENV = "RIFT_PVC_TCNN_WEIGHT_GRAD"
DEFAULT_STRATEGY = "bmm:256"


def strategy() -> str:
    value = os.environ.get(ENV, "").strip()
    return value or DEFAULT_STRATEGY


def weight_gradient(grad_out: torch.Tensor, x: torch.Tensor, how: str) -> torch.Tensor:
    """``d(sum(loss))/dW`` for ``y = x @ W.T``: ``grad_out.T @ x`` with the chosen reduction order."""
    if how == "gemm":
        return grad_out.t() @ x
    if how == "fp64":
        return (grad_out.double().t() @ x.double()).to(grad_out.dtype)
    kind, _, size = how.partition(":")
    block = int(size) if size else 256
    n, out_w, in_w = x.shape[0], grad_out.shape[1], x.shape[1]
    if kind == "bmm":
        pad = (-n) % block
        if pad:
            grad_out = torch.cat([grad_out, grad_out.new_zeros(pad, out_w)])
            x = torch.cat([x, x.new_zeros(pad, in_w)])
        blocks = grad_out.shape[0] // block
        partial = torch.bmm(grad_out.view(blocks, block, out_w).transpose(1, 2), x.view(blocks, block, in_w))
        return partial.float().sum(0).to(grad_out.dtype)
    if kind == "outer":
        acc = torch.zeros(out_w, in_w, dtype=torch.float32, device=x.device)
        for start in range(0, n, block):
            acc = acc + (grad_out[start:start + block, :, None].float() * x[start:start + block, None, :].float()).sum(0)
        return acc.to(grad_out.dtype)
    raise ValueError(f"{ENV} must be gemm, fp64, bmm:<block> or outer:<chunk>, got {how!r}")


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, how):
        ctx.save_for_backward(x, weight)
        ctx.how = how
        return x @ weight.t()

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        grad_x = grad_out @ weight if ctx.needs_input_grad[0] else None
        grad_w = weight_gradient(grad_out, x, ctx.how).to(weight.dtype) if ctx.needs_input_grad[1] else None
        return grad_x, grad_w, None


def linear(x: torch.Tensor, weight: torch.Tensor, how: str | None = None) -> torch.Tensor:
    """``x @ weight.T`` (``weight`` is ``[out, in]``) with the deterministic weight gradient."""
    how = strategy() if how is None else how
    if how == "gemm":
        return x @ weight.t()
    return _Linear.apply(x.contiguous(), weight.contiguous(), how)
