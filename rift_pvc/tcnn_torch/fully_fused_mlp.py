"""FullyFusedMLP: torch mirror of tiny-cuda-nn's fused MLP semantics.

Transcribed from the pinned ``src/fully_fused_mlp.cu`` (constructor,
``kernel_mlp_fused``, ``set_params_impl``, ``initialize_params``),
``src/network.cu`` (config defaults: ``n_hidden_layers`` 5, ``n_neurons`` 128,
``activation`` ReLU, ``output_activation`` None; widths 16/32/64/128),
``src/cpp_api.cu`` (a ``Network`` is a ``NetworkWithInputEncoding`` over an
``Identity`` encoding aligned to 16 whose padding entries are ``1``),
``include/tiny-cuda-nn/gpu_matrix.h`` (``initialize_xavier_uniform``) and
``common_device.h`` (``warp_activation`` formulas, ``K_ACT = 10``).

Forward: ``h = act(W_in x)``, ``(n_hidden_layers - 1)`` times ``h = act(W h)``,
``y = out_act(W_out h)`` truncated to ``n_output_dims`` (the fused kernel
pads the output to a multiple of 16). Weights are bias-free, row-major
``[out, in]`` and contiguous in the flat ``params`` in that order. In fp32
mode all matmuls are float32; in the fp16 parity mode the input and the
weights are cast to float16 and the matmuls run in float16 (the fused CUDA
kernel accumulates in fp16 tensor-core fragments; torch accumulates in fp32,
an accepted accumulation-order difference of the ledger, section 5).

Weight gradients use a fixed-order reduction (``linear.py``) because the
plain float32 GEMM is order-nondeterministic on PVC for the large K of a
training batch; tiny-cuda-nn's split-K reduction is fixed-order too.

Initialisation: Xavier-uniform per matrix, bound ``sqrt(6 / (fan_in +
fan_out))`` with fan_out = rows, fan_in = cols, values
``u * 2 * bound - bound`` drawn in matrix order from a torch CPU generator
seeded with ``seed``. ``Sine`` (SIREN) initialisation is not mirrored.
"""
from __future__ import annotations

import torch

from .layout import describe, mlp_layout
from .linear import linear, strategy as weight_grad_strategy
from .module import DEFAULT_SEED, ShimModule, compute_dtype, scale_grad

K_ACT = 10.0
_ACTIVATIONS = ("None", "ReLU", "LeakyReLU", "SiLU", "Exponential", "Sine", "Sigmoid", "Squareplus",
                "Softplus", "Tanh")


def canonical_activation(name: str) -> str:
    """``string_to_activation`` is case-insensitive."""
    key = str(name).lower()
    for candidate in _ACTIVATIONS:
        if candidate.lower() == key:
            return candidate
    raise ValueError(f"Invalid activation name: {name}")


def apply_activation(name: str, x: torch.Tensor) -> torch.Tensor:
    """``warp_activation`` formulas from common_device.h."""
    if name == "None":
        return x
    if name == "ReLU":
        return torch.relu(x)
    if name == "LeakyReLU":
        return x * torch.where(x > 0, torch.ones_like(x), torch.full_like(x, 0.01))
    if name == "SiLU":
        return x * torch.sigmoid(x)
    if name == "Exponential":
        return torch.exp(x)
    if name == "Sine":
        return torch.sin(x)
    if name == "Sigmoid":
        return torch.sigmoid(x)
    if name == "Squareplus":
        xk = x * K_ACT
        return 0.5 * (xk + torch.sqrt(xk * xk + 4.0)) / K_ACT
    if name == "Softplus":
        return torch.log(torch.exp(x * K_ACT) + 1.0) / K_ACT
    if name == "Tanh":
        return torch.tanh(x)
    raise ValueError(f"Invalid activation name: {name}")


class FullyFusedMLP(ShimModule):
    def __init__(self, n_input_dims: int, n_output_dims: int, network_config: dict, seed: int = DEFAULT_SEED):
        cfg = dict(network_config)
        otype = str(cfg.get("otype", "MLP"))
        if otype.lower() not in ("fullyfusedmlp", "megakernelmlp"):
            raise ValueError(f"torch shim mirrors FullyFusedMLP only, got network otype {otype!r}")
        self.activation = canonical_activation(cfg.get("activation", "ReLU"))
        self.output_activation = canonical_activation(cfg.get("output_activation", "None"))
        if self.activation == "Sine":
            raise NotImplementedError("torch shim does not mirror the SIREN initialisation of Sine networks")
        n_neurons = int(cfg.get("n_neurons", 128))
        n_hidden_layers = int(cfg.get("n_hidden_layers", 5))
        self.layout = mlp_layout(int(n_input_dims), int(n_output_dims), n_neurons, n_hidden_layers)
        super().__init__(n_input_dims, n_output_dims, seed, compute_dtype())
        self.network_config = cfg
        self.weight_grad = weight_grad_strategy()
        generator = self._generator()
        initial = torch.empty(self.layout.n_params, dtype=torch.float32)
        for matrix in self.layout.matrices:
            bound = matrix.xavier_bound
            uniform = torch.rand(matrix.numel, generator=generator, dtype=torch.float32)
            # gpu_matrix.h: rnd.next_float() * 2.0f * scale - scale
            initial[matrix.offset:matrix.offset + matrix.numel] = uniform * 2.0 * bound - bound
        self._set_params(initial)

    @property
    def padded_output_width(self) -> int:
        return self.layout.padded_output_width

    def hyperparams(self) -> dict:
        return {"otype": "FullyFusedMLP", "activation": self.activation, "output_activation": self.output_activation,
                "n_neurons": self.layout.n_neurons, "n_hidden_layers": self.layout.n_hidden_layers}

    def layout_description(self) -> dict:
        return describe(self.layout)

    def weight_matrices(self, params=None):
        """Views ``[out, in]`` into ``params`` in TCNN order (input, hidden..., output)."""
        params = self.params if params is None else params
        return [params[m.offset:m.offset + m.numel].view(m.rows, m.cols) for m in self.layout.matrices]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._check_input(x)
        pad = self.layout.padded_input_width - self.n_input_dims
        if pad:
            # identity.h: entries beyond num_to_encode are filled with 1.
            x = torch.cat([x, torch.ones(x.shape[0], pad, dtype=x.dtype, device=x.device)], dim=1)
        dtype = self.dtype
        half = dtype == torch.float16
        params = self.params
        if half:
            x = scale_grad(x, 1.0 / self.loss_scale)
            params = scale_grad(params, 1.0 / self.loss_scale)
        hidden = x.to(dtype)
        matrices = self.weight_matrices(params)
        how = self.weight_grad
        for index, weight in enumerate(matrices):
            hidden = linear(hidden, weight.to(dtype), how)   # deterministic weight gradient, see linear.py
            last = index == len(matrices) - 1
            hidden = apply_activation(self.output_activation if last else self.activation, hidden)
        out = hidden[:, :self.n_output_dims]
        return scale_grad(out, self.loss_scale) if half else out
