"""``tinycudann``-compatible pure-torch shim for the Radar Fields release on PVC.

Backend identity ``upstream-tcnn-torchshim`` (ledger D4,
``docs/RADAR_FIELDS_PVC_ADAPTATION.md``). The module is aliased as
``sys.modules["tinycudann"]`` by ``rift_pvc.radar_fields_upstream.install_shim``
only inside a PVC process (XPU backend, or ``RIFT_PVC_TCNN_SHIM=1`` for CPU
tests), before the authors' ``radarfields/nn/models.py`` executes
``import tinycudann as tcnn``. It is never a substitute for the real
tiny-cuda-nn in the CUDA environment.

Public surface (``bindings/torch/tinycudann/__init__.py``): ``Encoding``,
``Network``, ``NetworkWithInputEncoding``, ``free_temporary_memory``,
``supports_jit_fusion``. ``Encoding`` and ``Network`` are factories returning
the concrete shim modules; each is an ``nn.Module`` with exactly one flat
float32 ``params`` Parameter (so the release's ``get_params`` builds the same
five parameter groups it builds with tiny-cuda-nn) and the attributes
``n_input_dims``, ``n_output_dims``, ``seed``, ``dtype``, ``loss_scale``,
``native_tcnn_module`` (``None``).

Precision (ledger D3): float32 by default; ``RIFT_PVC_TCNN_HALF=1`` enables
the fp16 parity mode. ``PARITY`` records the outcome of the H100 replay test
(gate 2) and is copied into PVC checkpoints by the trainer twin.
"""
from __future__ import annotations

from .fully_fused_mlp import FullyFusedMLP
from .hashgrid import HashGridEncoding
from .layout import grid_layout, mlp_layout
from .linear import strategy as weight_grad_strategy
from .module import DEFAULT_SEED, compute_dtype, precision_mode
from .network_with_input_encoding import NetworkWithInputEncoding, build_encoding
from .spherical_harmonics import SphericalHarmonicsEncoding

__version__ = "torchshim-2"   # torchshim-1: plain-GEMM weight gradients (order-nondeterministic on PVC)
SHIM_VERSION = __version__
BACKEND_ID = "upstream-tcnn-torchshim"
TCNN_SOURCE_COMMIT = "749dd70c5afc5a9dadb85e5652ed65d55e0ba187"
# Numerical diagnostics, not a strict floating-point equivalence certificate.
# The user accepts lower-level arithmetic differences when model semantics match.
PARITY = {"job": 2153856, "status": "passed", "date": "2026-09-22", "test": "rift_pvc/tests/test_tcnn_parity.py",
          "notes": "H100 PCIe, campaign tiny-cuda-nn, jobs 2153856 + 2154357 (mean reduction, amplified records), PVC replay 2155072: "
                   "layout identical; vs fp32-precision TCNN twins grid 2.2e-5, SH 2.2e-8; release RadarField end-to-end alpha 1.2e-5, rd 5.1e-4; "
                   "vs fp16 modules forward <= 2.4e-3 at init scale (fp16 subnormal inputs) and <= 5.6e-4 on amplified records, "
                   "parameter gradients <= 3.2e-3 (amplified)"}


def Encoding(n_input_dims: int, encoding_config: dict, seed: int = DEFAULT_SEED, dtype=None):
    """``tinycudann.Encoding(n_input_dims, encoding_config, seed=1337, dtype=None)``."""
    return build_encoding(n_input_dims, encoding_config, seed=seed, dtype=dtype)


def Network(n_input_dims: int, n_output_dims: int, network_config: dict, seed: int = DEFAULT_SEED):
    """``tinycudann.Network(n_input_dims, n_output_dims, network_config, seed=1337)``."""
    return FullyFusedMLP(n_input_dims, n_output_dims, network_config, seed=seed)


def free_temporary_memory() -> None:
    """tiny-cuda-nn frees its GPU memory arena; the shim holds none."""


def supports_jit_fusion() -> bool:
    return False


def identity() -> dict:
    """Backend identity recorded by the PVC trainer twins."""
    return {"model_backend": BACKEND_ID, "version": SHIM_VERSION, "precision": precision_mode(),
            "weight_grad": weight_grad_strategy(), "fp16_loss_scale": 128.0 if precision_mode() == "fp16" else None,
            "tcnn_source_commit": TCNN_SOURCE_COMMIT, "parity": dict(PARITY)}


__all__ = ["Encoding", "Network", "NetworkWithInputEncoding", "free_temporary_memory", "supports_jit_fusion",
           "HashGridEncoding", "SphericalHarmonicsEncoding", "FullyFusedMLP", "grid_layout", "mlp_layout",
           "precision_mode", "compute_dtype", "identity", "SHIM_VERSION", "BACKEND_ID", "PARITY", "__version__"]
