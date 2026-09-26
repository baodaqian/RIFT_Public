"""tiny-cuda-nn parameter layout, mirrored from the pinned source (no inference).

Every rule here is transcribed from the tiny-cuda-nn checkout pinned by the
campaign (commit ``749dd70c5afc5a9dadb85e5652ed65d55e0ba187``, snapshot
``.../h100_smoke_20260920_6d58645/code/external/tiny-cuda-nn``):

* ``include/tiny-cuda-nn/encodings/grid.h`` ``GridEncodingTemplated`` constructor:
  per-level resolution from ``grid_resolution(grid_scale(...))``, ``params_in_level``
  padded to a multiple of 8, capped at ``1 << log2_hashmap_size`` for hash grids,
  the ``m_offset_table`` prefix sums, ``n_params = offset_table[n_levels] *
  N_FEATURES_PER_LEVEL`` (features interleaved per vertex).
* ``include/tiny-cuda-nn/common_device.h`` ``grid_scale`` (``exp2f(level *
  log2_per_level_scale) * base_resolution - 1.0f``, float32 arithmetic),
  ``grid_resolution`` (``ceilf(scale) + 1``) and ``grid_index`` (dense stride
  indexing while ``hashmap_size >= resolution**dims``, otherwise the coherent
  prime hash, then ``% hashmap_size``; ``MAX_BASES`` overflow guard).
* ``src/fully_fused_mlp.cu`` constructor / ``set_params_impl`` /
  ``initialize_params``: weight matrices ``[n_neurons, input_width]``,
  ``(n_hidden_layers - 1) x [n_neurons, n_neurons]``, ``[padded_output_width,
  n_neurons]`` with ``padded_output_width = next_multiple(output_width, 16)``,
  stored contiguously and row-major (``[out, in]``), bias-free, Xavier-uniform
  ``sqrt(6 / (fan_in + fan_out))`` per matrix in that order.
* ``src/cpp_api.cu``: ``create_network`` is ``create_network_with_input_encoding``
  over an ``Identity`` encoding; ``network_with_input_encoding.h`` aligns the
  encoding output to ``minimum_alignment(network)`` (16 for FullyFusedMLP) so the
  MLP input width is ``next_multiple(n_input_dims, 16)``; ``encodings/identity.h``
  fills the padding with ``1``; the flat ``params`` vector holds the network
  parameters first, then the encoding's. Standalone ``Encoding`` objects are
  created with alignment 0 (``src/encoding.cu``), i.e. no output padding.

The float32 evaluation of ``grid_scale`` is reproduced with numpy float32 so the
discrete per-level resolutions (and therefore the table sizes and offsets) are
the ones the CUDA library computes. For the production configuration (16
levels, base 16, ``per_level_scale = 2**(1/3)``) the products
``level * log2(per_level_scale)`` round to exact integers in float32 at levels
0, 3, ..., 15, so the layout does not depend on the last-ulp behaviour of
``log2f``/``exp2f``; the parity test still asserts ``n_params`` against the
H100 dump.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

# common_device.h: coherent_prime_hash factors (dimension i uses factors[i]).
COHERENT_PRIMES = (1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737)
# common_device.h grid_index: largest resolution per N_DIMS whose dense stride fits uint32.
MAX_BASES = (0x0, 0xFFFFFFFF, 0xFFFF, 0x659, 0xFF, 0x54, 0x28, 0x17, 0xF, 0xB, 0x9)
MAX_N_LEVELS = 128          # multi_level_interface.h
UINT32_MAX = 0xFFFFFFFF
GRID_ALIGNMENT = 8          # grid.h: next_multiple(params_in_level, 8u)
MLP_ALIGNMENT = 16          # fully_fused_mlp.h REQUIRED_ALIGNMENT (16x16x16 tensor ops)
MLP_WIDTHS = (16, 32, 64, 128)   # network.cu: FullyFusedMLP instantiations
GRID_TYPES = ("Hash", "Dense", "Tiled")


def next_multiple(value: int, divisor: int) -> int:
    """``common.h`` ``next_multiple``: round ``value`` up to a multiple of ``divisor``."""
    return ((int(value) + divisor - 1) // divisor) * divisor


def f32(value) -> np.float32:
    return np.float32(value)


def log2_per_level_scale(per_level_scale: float) -> np.float32:
    """``std::log2(m_per_level_scale)`` with ``m_per_level_scale`` a float."""
    scale32 = f32(per_level_scale)             # json double -> float member
    return f32(math.log2(float(scale32)))      # log2f, correctly rounded via double


def grid_scale(level: int, log2_scale: np.float32, base_resolution: int) -> np.float32:
    """``grid_scale``: ``exp2f(level * log2_per_level_scale) * base_resolution - 1.0f``."""
    exponent = f32(f32(level) * f32(log2_scale))          # uint32 * float -> float
    power = f32(2.0 ** float(exponent))                   # exp2f (exact for integers)
    return f32(f32(power * f32(base_resolution)) - f32(1.0))


def grid_resolution(scale: np.float32) -> int:
    """``grid_resolution``: ``(uint32_t)ceilf(scale) + 1`` (vertex count)."""
    return int(math.ceil(float(scale))) + 1


@dataclass(frozen=True)
class GridLevel:
    level: int
    scale: float          # float32 value used by the kernels (pos = fma(scale, x, 0.5))
    resolution: int       # vertices per axis
    params_in_level: int  # table entries (vertices, padded to 8, capped by the hash map)
    offset: int           # entry offset of this level in the table (features interleaved)
    hashed: bool          # grid_index uses the coherent prime hash at this level


@dataclass(frozen=True)
class GridLayout:
    n_pos_dims: int
    n_levels: int
    n_features_per_level: int
    log2_hashmap_size: int
    base_resolution: int
    per_level_scale: float        # float32 member value
    log2_scale: float             # float32 log2 of it
    grid_type: str
    levels: tuple
    n_params: int

    def slices(self):
        """``{name: (offset, shape)}`` into the flat ``params`` vector."""
        f = self.n_features_per_level
        return {f"level{lv.level}": (lv.offset * f, (lv.params_in_level, f)) for lv in self.levels}


def grid_layout(n_pos_dims: int, n_levels: int, n_features_per_level: int, log2_hashmap_size: int,
                base_resolution: int, per_level_scale: float, grid_type: str = "Hash") -> GridLayout:
    """``GridEncodingTemplated`` constructor, grid.h."""
    if grid_type not in GRID_TYPES:
        raise ValueError(f"GridEncoding: invalid grid type {grid_type}")
    if n_levels > MAX_N_LEVELS:
        raise ValueError(f"GridEncoding: m_n_levels={n_levels} must be at most MAX_N_LEVELS={MAX_N_LEVELS}")
    if n_pos_dims < 1 or n_pos_dims >= len(MAX_BASES):
        raise ValueError("grid_index can only be used for N_DIMS <= 10")
    scale32 = f32(per_level_scale)
    log2_scale = log2_per_level_scale(per_level_scale)
    max_params = UINT32_MAX // 2
    levels, offset = [], 0
    for level in range(n_levels):
        scale = grid_scale(level, log2_scale, base_resolution)
        resolution = grid_resolution(scale)
        # std::pow((float)resolution, N_POS_DIMS) > (float)max_params ? max_params : powi(resolution, N_POS_DIMS)
        if float(resolution) ** n_pos_dims > float(f32(max_params)):
            params_in_level = max_params
        else:
            params_in_level = resolution ** n_pos_dims
        params_in_level = next_multiple(params_in_level, GRID_ALIGNMENT)
        if grid_type == "Tiled":
            params_in_level = min(params_in_level, base_resolution ** n_pos_dims)
        elif grid_type == "Hash":
            params_in_level = min(params_in_level, 1 << log2_hashmap_size)
        stride = resolution ** n_pos_dims if resolution <= MAX_BASES[n_pos_dims] else UINT32_MAX
        hashed = grid_type == "Hash" and params_in_level < stride
        levels.append(GridLevel(level, float(scale), resolution, params_in_level, offset, hashed))
        offset += params_in_level
    return GridLayout(n_pos_dims, n_levels, n_features_per_level, log2_hashmap_size, base_resolution,
                      float(scale32), float(log2_scale), grid_type, tuple(levels),
                      offset * n_features_per_level)


@dataclass(frozen=True)
class MLPMatrix:
    name: str
    offset: int
    rows: int   # fan_out (m)
    cols: int   # fan_in (n); row-major [rows, cols]

    @property
    def numel(self) -> int:
        return self.rows * self.cols

    @property
    def xavier_bound(self) -> float:
        """``initialize_xavier_uniform``: ``sqrt(6.0f / (fan_in + fan_out))`` (float32)."""
        return float(f32(math.sqrt(float(f32(6.0) / f32(self.rows + self.cols)))))


@dataclass(frozen=True)
class MLPLayout:
    n_input_dims: int
    padded_input_width: int
    n_neurons: int
    n_hidden_layers: int
    n_output_dims: int
    padded_output_width: int
    matrices: tuple
    n_params: int

    def slices(self):
        return {m.name: (m.offset, (m.rows, m.cols)) for m in self.matrices}


def mlp_layout(n_input_dims: int, n_output_dims: int, n_neurons: int, n_hidden_layers: int) -> MLPLayout:
    """``FullyFusedMLP`` constructor (fully_fused_mlp.cu) behind ``create_network``'s
    Identity encoding aligned to 16 (network_with_input_encoding.h)."""
    if n_hidden_layers <= 0:
        raise ValueError("FullyFusedMLP requires at least 1 hidden layer (3 layers in total).")
    if n_neurons not in MLP_WIDTHS:
        raise ValueError(f"FullyFusedMLP only supports 16, 32, 64, and 128 neurons, but got {n_neurons}.")
    if n_input_dims < 1 or n_output_dims < 1:
        raise ValueError("FullyFusedMLP needs positive input and output widths")
    padded_input = next_multiple(n_input_dims, MLP_ALIGNMENT)
    padded_output = next_multiple(n_output_dims, MLP_ALIGNMENT)
    matrices, offset = [], 0
    matrices.append(MLPMatrix("input", offset, n_neurons, padded_input))
    offset += n_neurons * padded_input
    for k in range(n_hidden_layers - 1):
        matrices.append(MLPMatrix(f"hidden{k}", offset, n_neurons, n_neurons))
        offset += n_neurons * n_neurons
    matrices.append(MLPMatrix("output", offset, padded_output, n_neurons))
    offset += padded_output * n_neurons
    return MLPLayout(n_input_dims, padded_input, n_neurons, n_hidden_layers, n_output_dims, padded_output,
                     tuple(matrices), offset)


def describe(layout) -> dict:
    """JSON-serialisable summary for logs, checkpoints and the parity dump."""
    if isinstance(layout, GridLayout):
        return {
            "kind": "grid", "n_pos_dims": layout.n_pos_dims, "n_levels": layout.n_levels,
            "n_features_per_level": layout.n_features_per_level, "log2_hashmap_size": layout.log2_hashmap_size,
            "base_resolution": layout.base_resolution, "per_level_scale_f32": layout.per_level_scale,
            "log2_per_level_scale_f32": layout.log2_scale, "grid_type": layout.grid_type, "n_params": layout.n_params,
            "levels": [{"level": lv.level, "scale": lv.scale, "resolution": lv.resolution,
                        "params_in_level": lv.params_in_level, "offset": lv.offset, "hashed": lv.hashed}
                       for lv in layout.levels],
        }
    if isinstance(layout, MLPLayout):
        return {
            "kind": "mlp", "n_input_dims": layout.n_input_dims, "padded_input_width": layout.padded_input_width,
            "n_neurons": layout.n_neurons, "n_hidden_layers": layout.n_hidden_layers,
            "n_output_dims": layout.n_output_dims, "padded_output_width": layout.padded_output_width,
            "n_params": layout.n_params,
            "matrices": [{"name": m.name, "offset": m.offset, "rows": m.rows, "cols": m.cols,
                          "xavier_bound": m.xavier_bound} for m in layout.matrices],
        }
    raise TypeError(f"unknown layout {type(layout).__name__}")
