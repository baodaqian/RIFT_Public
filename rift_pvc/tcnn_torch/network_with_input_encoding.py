"""``NetworkWithInputEncoding``: encoding feeding a FullyFusedMLP with one flat
``params`` (network parameters first, then the encoding's, as
``network_with_input_encoding.h`` ``set_params_impl`` lays them out).

The encoding output is aligned to the network's minimum alignment (16). The
padding value follows the pinned kernels with JIT disabled: ``HashGrid`` pads
with zeros after its features (``grid.h`` ``forward_impl``),
``SphericalHarmonics`` writes its ``num_to_pad`` ones *before* the
coefficients (``kernel_sh``). Not used by the Radar Fields release (which
builds standalone ``Encoding`` and ``Network`` objects); provided for API
completeness and covered by the unit tests only.
"""
from __future__ import annotations

import torch
from torch.func import functional_call

from .fully_fused_mlp import FullyFusedMLP
from .hashgrid import HashGridEncoding
from .layout import MLP_ALIGNMENT, next_multiple
from .module import DEFAULT_SEED, ShimModule
from .spherical_harmonics import SphericalHarmonicsEncoding


def build_encoding(n_input_dims: int, encoding_config: dict, seed: int = DEFAULT_SEED, dtype=None):
    otype = str(dict(encoding_config).get("otype", "OneBlob"))
    key = otype.lower()
    if key in ("hashgrid", "grid", "densegrid", "tiledgrid"):
        return HashGridEncoding(n_input_dims, encoding_config, seed=seed, dtype=dtype)
    if key == "sphericalharmonics":
        return SphericalHarmonicsEncoding(n_input_dims, encoding_config, seed=seed, dtype=dtype)
    raise ValueError(f"torch shim mirrors HashGrid and SphericalHarmonics encodings only, got otype {otype!r}")


class NetworkWithInputEncoding(ShimModule):
    def __init__(self, n_input_dims: int, n_output_dims: int, encoding_config: dict, network_config: dict,
                 seed: int = DEFAULT_SEED):
        encoding = build_encoding(n_input_dims, encoding_config, seed=seed)
        padded = next_multiple(encoding.n_output_dims, MLP_ALIGNMENT)
        network = FullyFusedMLP(padded, n_output_dims, network_config, seed=seed)
        super().__init__(n_input_dims, n_output_dims, seed, network.dtype)
        self.encoding_config, self.network_config = dict(encoding_config), dict(network_config)
        # Plain attributes: the combined module owns the single flat Parameter.
        object.__setattr__(self, "_encoding", encoding)
        object.__setattr__(self, "_network", network)
        self.n_pad = padded - encoding.n_output_dims
        self.network_offset, self.encoding_offset = 0, network.params.numel()
        self._set_params(torch.cat([network.params.detach(), encoding.params.detach()]))
        del network.params, encoding.params   # the sub-modules run through functional_call

    def hyperparams(self) -> dict:
        return {"otype": "NetworkWithInputEncoding", "encoding": self._encoding.hyperparams(),
                "network": self._network.hyperparams()}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._check_input(x)
        network_params = self.params[:self.encoding_offset]
        encoding_params = self.params[self.encoding_offset:]
        encoded = functional_call(self._encoding, {"params": encoding_params}, (x,))
        if self.n_pad:
            pad = torch.ones(x.shape[0], self.n_pad, dtype=encoded.dtype, device=encoded.device)
            if isinstance(self._encoding, SphericalHarmonicsEncoding):
                encoded = torch.cat([pad, encoded], dim=1)          # kernel_sh: ones first
            else:
                encoded = torch.cat([encoded, torch.zeros_like(pad)], dim=1)
        return functional_call(self._network, {"params": network_params}, (encoded,))
