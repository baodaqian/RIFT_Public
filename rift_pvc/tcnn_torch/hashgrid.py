"""HashGrid encoding: torch mirror of tiny-cuda-nn ``GridEncodingTemplated``.

Transcribed from the pinned ``include/tiny-cuda-nn/encodings/grid.h``
(``kernel_grid``, constructor, ``initialize_params``) and
``common_device.h`` (``pos_fract``, ``grid_index``, ``coherent_prime_hash``):

* ``pos = fmaf(scale, x, 0.5f)``; ``pos_grid = (uint32_t)(int)floorf(pos)``;
  ``frac = pos - floor`` (0.5 stagger, no clamping, wraparound at the edge as
  in the CUDA kernel);
* corner ``idx`` (bit ``dim`` set means ``pos_grid[dim] + 1``) with weight
  ``prod(bit ? frac : 1 - frac)``;
* dense stride index ``sum(pos_grid[dim] * resolution**dim)`` in uint32 while
  the level's table holds all vertices (``hashmap_size >= resolution**dims``),
  otherwise ``xor(pos_grid[dim] * prime[dim])`` in uint32 with the coherent
  primes ``(1, 2654435761, 805459861, ...)``; then ``% hashmap_size``;
* ``N_FEATURES_PER_LEVEL`` features interleaved per vertex, output ordered
  level-major (``encoded[level * F + f]``), no output padding for a standalone
  encoding (alignment 0);
* parameters initialised uniformly in ``[-1e-4, 1e-4)`` (``grid.h``
  ``initialize_params``), from a torch CPU generator seeded with ``seed``
  (the pcg32 stream is not reproduced).

Only what the release uses is mirrored: ``Linear`` interpolation, no
stochastic interpolation, no fixed-point positions, ``CoherentPrime`` hashing
(the pinned build compiles no other hash). Anything else raises. In fp32 mode
the trilinear sum is a float32 reduction; in the fp16 parity mode
(``RIFT_PVC_TCNN_HALF=1``) parameters and weights are cast to float16 and the
eight corners are accumulated in the kernel's order with fused
multiply-adds, like ``fma((T)weight, grid_val, result)`` with ``T = __half``.
Parameter gradients are accumulated in float32 by autograd (the CUDA build
accumulates them in fp16 atomics for two features per level).
"""
from __future__ import annotations

import math

import torch

from .layout import COHERENT_PRIMES, UINT32_MAX, describe, grid_layout
from .module import DEFAULT_SEED, ShimModule, resolve_dtype, scale_grad

_GRID_OTYPES = {"hashgrid": "Hash", "grid": "Hash", "densegrid": "Dense", "tiledgrid": "Tiled"}


class HashGridEncoding(ShimModule):
    def __init__(self, n_input_dims: int, encoding_config: dict, seed: int = DEFAULT_SEED, dtype=None):
        cfg = dict(encoding_config)
        otype = str(cfg.get("otype", "Grid"))
        default_type = _GRID_OTYPES.get(otype.lower(), "Hash")
        grid_type = str(cfg.get("type", default_type))
        features = int(cfg.get("n_features_per_level", 2))
        if features not in (1, 2, 4, 8):
            raise ValueError("GridEncoding: n_features_per_level must be 1, 2, 4, or 8.")
        if "n_features" in cfg or "n_grid_features" in cfg:
            n_features = int(cfg["n_features"] if "n_features" in cfg else cfg["n_grid_features"])
            if "n_levels" in cfg:
                raise ValueError("GridEncoding: may not specify n_features and n_levels simultaneously (one determines the other)")
        else:
            n_features = features * int(cfg.get("n_levels", 16))
        if n_features % features:
            raise ValueError(f"GridEncoding: n_features={n_features} must be a multiple of N_FEATURES_PER_LEVEL={features}")
        n_levels = n_features // features
        log2_hashmap_size = int(cfg.get("log2_hashmap_size", 19))
        base_resolution = int(cfg.get("base_resolution", 16))
        if grid_type == "Dense":
            default_scale = math.exp(math.log(256.0 / base_resolution) / max(n_levels - 1, 1))
        else:
            default_scale = 2.0
        per_level_scale = float(cfg.get("per_level_scale", default_scale))
        interpolation = str(cfg.get("interpolation", "Linear"))
        if interpolation.lower() != "linear":
            raise NotImplementedError(f"torch shim mirrors Linear grid interpolation only, got {interpolation!r}")
        if cfg.get("stochastic_interpolation", False):
            raise NotImplementedError("torch shim does not mirror stochastic_interpolation")
        if cfg.get("fixed_point_pos", False):
            raise NotImplementedError("torch shim does not mirror fixed_point_pos")
        hash_type = str(cfg.get("hash", "CoherentPrime"))
        if hash_type.lower() != "coherentprime":
            raise ValueError(f"GridEncoding: compiled without {hash_type} hash support.")
        if n_input_dims not in (2, 3, 4):
            raise ValueError("GridEncoding: number of input dims must be 2 or 3.")
        self.layout = grid_layout(n_input_dims, n_levels, features, log2_hashmap_size, base_resolution,
                                  per_level_scale, grid_type)
        super().__init__(n_input_dims, n_features, seed, resolve_dtype(dtype))
        self.encoding_config = cfg
        self.grid_type = grid_type
        # grid.h initialize_params: uniform in [-1e-4 * scale, 1e-4 * scale) with scale = 1.
        uniform = torch.rand(self.layout.n_params, generator=self._generator(), dtype=torch.float32)
        self._set_params(uniform * 2.0e-4 - 1.0e-4)
        corners = torch.tensor([[(idx >> dim) & 1 for dim in range(n_input_dims)]
                                for idx in range(1 << n_input_dims)], dtype=torch.int64)
        self.register_buffer("corner_bits", corners, persistent=False)
        self.register_buffer("primes", torch.tensor(COHERENT_PRIMES[:n_input_dims], dtype=torch.int64),
                             persistent=False)

    def hyperparams(self) -> dict:
        result = {"otype": "Grid", "type": self.grid_type, "n_levels": self.layout.n_levels,
                  "n_features_per_level": self.layout.n_features_per_level,
                  "base_resolution": self.layout.base_resolution, "per_level_scale": self.layout.per_level_scale,
                  "interpolation": "Linear", "hash": "CoherentPrime"}
        if self.grid_type == "Hash":
            result["log2_hashmap_size"] = self.layout.log2_hashmap_size
        return result

    def layout_description(self) -> dict:
        return describe(self.layout)

    def level_index(self, corners: torch.Tensor, level) -> torch.Tensor:
        """``grid_index`` for integer corner coordinates ``[..., D]`` (uint32 semantics)."""
        dims = self.n_input_dims
        if level.hashed:
            index = torch.zeros(corners.shape[:-1], dtype=torch.int64, device=corners.device)
            for dim in range(dims):
                index = index ^ ((corners[..., dim] * int(COHERENT_PRIMES[dim])) & UINT32_MAX)
        else:
            index = torch.zeros(corners.shape[:-1], dtype=torch.int64, device=corners.device)
            stride = 1
            for dim in range(dims):
                index = index + corners[..., dim] * stride
                stride *= level.resolution
            index = index & UINT32_MAX
        return torch.remainder(index, level.params_in_level)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._check_input(x)
        count = x.shape[0]
        features = self.layout.n_features_per_level
        half = self.dtype == torch.float16
        params = self.params
        if half:
            # loss scaling as in the fp16 build: gradients inside the half ops are 128x larger
            x = scale_grad(x, 1.0 / self.loss_scale)
            params = scale_grad(params, 1.0 / self.loss_scale)
        table = params.view(-1, features)
        if half:
            table = table.to(torch.float16)
        x64 = x.to(torch.float64)
        bits = self.corner_bits.to(x.device)
        bits_mask = bits.to(torch.bool)[None]          # [1, C, D]
        outputs = []
        for level in self.layout.levels:
            # pos_fract: fmaf(scale, input, 0.5f) as one float32 rounding of the exact product-sum.
            pos = (x64 * level.scale + 0.5).to(torch.float32)
            floor = torch.floor(pos)
            frac = pos - floor
            base = floor.to(torch.int64)
            corners = base[:, None, :] + bits[None]     # [N, C, D]
            index = self.level_index(corners, level)
            feats = table[level.offset + index]         # [N, C, F]
            weights = torch.where(bits_mask, frac[:, None, :], 1.0 - frac[:, None, :]).prod(-1)  # [N, C]
            if half:
                acc = torch.zeros(count, features, dtype=torch.float16, device=x.device)
                for corner in range(bits.shape[0]):
                    acc = torch.addcmul(acc, weights[:, corner:corner + 1].to(torch.float16), feats[:, corner])
                outputs.append(acc)
            else:
                outputs.append((weights[..., None] * feats).sum(1))
        out = torch.cat(outputs, dim=-1)
        return scale_grad(out, self.loss_scale) if half else out
