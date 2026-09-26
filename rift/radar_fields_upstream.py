"""Load the original pinned Radar Fields implementation without modifying it.

The checkout is deliberately an explicit dependency of the audited recipe.
No download/install or fallback happens at runtime. The portable Torch model
is a separately recorded backend, never substituted for the official TCNN net.
"""

from functools import lru_cache
import importlib
from pathlib import Path
import sys

import torch
from torch import nn


REFERENCE_ROOT = Path(__file__).resolve().parents[1] / "external" / "RadarFields_reference"
SOURCE_FILES = (
    "radarfields/nn/pose_refinement.py", "utils/data.py", "utils/vis.py",
    "radarfields/figures.py", "configs/radarfields.ini", "main.py", "parse.py",
    "radarfields/train.py", "radarfields/sampler.py", "radarfields/dataset.py",
    "radarfields/radar.py", "radarfields/nn/models.py", "radarfields/nn/tcnn_utils.py",
    "utils/train.py", "radarfields/nn/encoding.py",
)


def verify_sources():
    """Check dependency availability; GitHub source is trusted without hashing."""
    for relative in SOURCE_FILES:
        source = REFERENCE_ROOT / relative
        if not source.is_file():
            raise RuntimeError(f"Radar Fields needs the upstream source file: {source}")


@lru_cache(maxsize=None)
def original_module(name):
    verify_sources()
    for prefix in ("radarfields", "utils"):
        existing = sys.modules.get(prefix)
        if existing is not None:
            location = getattr(existing, "__file__", None)
            if location is None or not Path(location).resolve().is_relative_to(REFERENCE_ROOT.resolve()):
                raise RuntimeError(f"module {prefix} conflicts with the pinned Radar Fields checkout")
    sys.path.insert(0, str(REFERENCE_ROOT))
    try:
        module = importlib.import_module(name)
    finally:
        sys.path.pop(0)
    if not Path(module.__file__).resolve().is_relative_to(REFERENCE_ROOT.resolve()):
        raise RuntimeError("loaded an unpinned Radar Fields module")
    return module


def check_model_backend(args):
    verify_sources()
    if args.model_backend == "upstream-tcnn":
        if torch.device(args.device).type != "cuda":
            raise ValueError("upstream-tcnn requires CUDA; select --model-backend torch explicitly for CPU checks")
        if not torch.cuda.is_available():
            raise RuntimeError("upstream-tcnn requires an allocated CUDA device")
        try:
            importlib.import_module("tinycudann")
        except ImportError as exc:
            raise RuntimeError("upstream-tcnn requires tiny-cuda-nn in the allocated environment; no automatic model substitution") from exc
        for key, expected in {"sh_degree": 3, "hash_features": 2,
                              "hash_base_resolution": 16, "hash_log2_size": 19}.items():
            if getattr(args, key) != expected:
                raise ValueError(f"upstream-tcnn fixes {key}={expected}")


class OriginalRadarFieldsModel(nn.Module):
    """Original RadarField + metric-to-unit input and RIFT readout interfaces.

    Neural architecture, TCNN encoding, direction inputs, and activations are
    the original implementation. This wrapper does not correct or replace the
    release's SH conventions. CPU fallback has a distinct checkpoint recipe.
    """

    def __init__(self, args):
        super().__init__()
        check_model_backend(args)
        self.extent = float(args.extent)
        self.source_adapted = getattr(args, "recipe", None) == "source-adapted-v3"
        model_class = original_module("radarfields.nn.models").RadarField
        self.original = model_class(
            in_dim=3, xyz_encoding="HashGrid", num_layers=4,
            hidden_dim=args.hidden_dim, xyz_feat_dim=args.feature_dim,
            alpha_dim=1, alpha_activation="sigmoid", sigmoid_tightness=args.sigmoid_tightness,
            rd_dim=1, softplus_rd=True, angle_dim=3, angle_in_layer=3,
            angle_encoding="SphericalHarmonics", resolution=args.hash_final_resolution,
            n_levels=args.hash_levels, bound=1, bn=not args.no_batch_norm, use_tcnn=True,
        )

    def forward(self, xyz, view_direction, mask_progress=None):
        if not torch.isfinite(xyz).all() or (xyz.abs() > self.extent).any():
            raise ValueError("original RF queries must be inside the registered support")
        directions = view_direction if self.source_adapted else torch.nn.functional.normalize(view_direction, dim=-1)
        if directions.ndim == 1:
            directions = directions[None, :].expand_as(xyz)
        output = self.original((xyz + self.extent) / (2 * self.extent), directions,
                               sin_epoch=mask_progress)
        alpha, rho = output["alpha"].reshape(-1), output["rd"].reshape(-1)
        if not self.source_adapted:
            alpha, rho = alpha.float(), rho.float()
        return {"alpha": alpha, "reflectance": rho, "rcs": alpha * rho}

    def get_params(self, lr):
        return self.original.get_params(lr)

    def query_chunked(self, xyz, view_direction, mask_progress=None, chunk_size=32768):
        if chunk_size < 1 or len(xyz) < 1:
            raise ValueError("query batch/chunk must be nonempty")
        if self.training:
            # Upstream BN sees a neural-query batch, not independent chunks.
            norms = [m for m in self.original.modules() if isinstance(m, nn.BatchNorm1d)]
            norm_training = [norm.training for norm in norms]
            if len(xyz) == 1 and not self.source_adapted:
                for norm in norms:
                    norm.eval()
            try:
                return self(xyz, view_direction, mask_progress)
            finally:
                for norm, training in zip(norms, norm_training):
                    norm.train(training)
        result = {key: [] for key in ("alpha", "reflectance", "rcs")}
        for start in range(0, len(xyz), chunk_size):
            direction = view_direction[start:start+chunk_size] if view_direction.ndim == 2 else view_direction
            output = self(xyz[start:start+chunk_size], direction, mask_progress)
            for key in result:
                result[key].append(output[key])
        return {key: torch.cat(value) for key, value in result.items()}
