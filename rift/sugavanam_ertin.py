"""Sugavanam--Ertin scattering-centre to neural-SDF baseline.

The paper (arXiv:2602.17556) is a two-stage method:

1. solve a sparsity-regularized coherent SAR inverse problem and threshold the
   joint scattering magnitude into a point cloud; and
2. denoise that cloud with a Fourier-feature coordinate MLP representing a
   signed distance function (SDF).

RIFT's B787 adapter uses its validated bistatic range operator for stage 1,
because the dataset is near-field MIMO rather than the paper's monostatic
far-field SAR.  ``train.py --scene-repr grid --l1-weight ...`` produces that
stage's checkpoint.  This module implements the paper-specific boundary
between stages and the SDF/iso-point machinery.  Ground-truth geometry is
deliberately absent from every function in this file.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn


PAPER_ID = "arXiv:2602.17556"
METHOD_NAME = "Sugavanam--Ertin neural SDF reimplementation"


@dataclass
class ScatteringCloud:
    points: np.ndarray
    normals: np.ndarray
    magnitude: np.ndarray
    threshold: float
    source_epoch: int
    source_checkpoint: str
    extent: float
    granularity: int


class FourierFeatureSDF(nn.Module):
    """Eight-layer, width-512 Softplus SDF MLP with the paper's layer-4 skip.

    ``n_fourier`` is the paper's N_F.  Coordinates are divided by ``extent``
    before the fixed random Fourier map.  The paper does not publish its
    Fourier scale or loss weights, so both are explicit experiment parameters
    and are serialized in every checkpoint.
    """

    def __init__(
        self,
        extent: float,
        n_fourier: int = 9,
        fourier_scale: float = 2.0,
        hidden_dim: int = 512,
        n_layers: int = 8,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if extent <= 0:
            raise ValueError("extent must be positive")
        if n_fourier <= 0 or hidden_dim <= 0 or n_layers < 5:
            raise ValueError("need n_fourier>0, hidden_dim>0, and n_layers>=5")
        self.extent = float(extent)
        self.n_fourier = int(n_fourier)
        self.fourier_scale = float(fourier_scale)
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.seed = int(seed)

        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        bands = torch.randn(n_fourier, 3, generator=gen) * fourier_scale
        self.register_buffer("fourier_bands", bands)
        encoded_dim = 3 + 2 * n_fourier

        layers = []
        for i in range(n_layers):
            in_dim = encoded_dim if i == 0 else hidden_dim
            if i == 4:
                in_dim += encoded_dim
            layer = nn.Linear(in_dim, hidden_dim)
            nn.init.normal_(layer.weight, mean=0.0, std=math.sqrt(2.0 / in_dim))
            nn.init.zeros_(layer.bias)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)
        self.output = nn.Linear(hidden_dim, 1)
        nn.init.normal_(self.output.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.output.bias)
        self.activation = nn.Softplus(beta=100.0)

    def encode(self, xyz: torch.Tensor) -> torch.Tensor:
        x = xyz / self.extent
        phase = 2.0 * math.pi * (x @ self.fourier_bands.T)
        return torch.cat((x, torch.sin(phase), torch.cos(phase)), dim=-1)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        encoded = self.encode(xyz)
        h = encoded
        for i, layer in enumerate(self.layers):
            if i == 4:
                h = torch.cat((h, encoded), dim=-1)
            h = self.activation(layer(h))
        # Metric-valued, bounded SDF.  The paper specifies a tanh output.
        return self.extent * torch.tanh(self.output(h)).squeeze(-1)

    def config(self) -> Dict[str, object]:
        return {
            "extent": self.extent,
            "n_fourier": self.n_fourier,
            "fourier_scale": self.fourier_scale,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "seed": self.seed,
        }


def _torch_load(path: str, map_location: str = "cpu") -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch < 2.6
        return torch.load(path, map_location=map_location)


def estimate_pca_normals(
    points: np.ndarray,
    radius: float,
    min_neighbors: int = 8,
    max_neighbors: int = 32,
) -> np.ndarray:
    """Local-plane normals, with nearest-neighbour fallback for sparse points.

    The paper uses a 0.3 m radius on vehicle-scale CVDomes scenes.  Our CLI
    defaults to three stage-1 voxel pitches so the neighbourhood scales to the
    10 cm B787 rather than copying a dimensionful constant blindly.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 3:
        raise ValueError("points must have shape [N,3] with N>=3")
    tree = cKDTree(pts)
    centre = np.median(pts, axis=0)
    normals = np.empty_like(pts)
    k_fallback = min(max(min_neighbors, 3), len(pts))
    for i, p in enumerate(pts):
        idx = tree.query_ball_point(p, radius)
        if len(idx) < 3:
            _, idx = tree.query(p, k=k_fallback)
            idx = np.atleast_1d(idx).tolist()
        elif len(idx) > max_neighbors:
            _, idx = tree.query(p, k=max_neighbors)
            idx = np.atleast_1d(idx).tolist()
        local = pts[np.asarray(idx)]
        centred = local - local.mean(axis=0, keepdims=True)
        cov = centred.T @ centred / max(len(local), 1)
        _, vec = np.linalg.eigh(cov)
        n = vec[:, 0]
        # The paper's cosine loss is sign invariant.  Orienting outward only
        # makes exported normals deterministic and does not change training.
        if np.dot(n, p - centre) < 0:
            n = -n
        normals[i] = n / max(np.linalg.norm(n), 1e-12)
    return normals.astype(np.float32)


def load_scattering_cloud(
    checkpoint_path: str,
    threshold_fraction: float = 0.15,
    max_points: int = 20000,
    normal_radius: Optional[float] = None,
) -> ScatteringCloud:
    """Threshold an isotropic RIFT grid checkpoint into the paper's P set.

    The threshold is relative to the maximum active magnitude, mirroring Eq. 6
    (the source paper does not publish tau).  ``max_points`` retains the
    strongest responses only when needed to keep PCA and SDF batches bounded.
    """
    if not (0.0 < threshold_fraction < 1.0):
        raise ValueError("threshold_fraction must be in (0,1)")
    ck = _torch_load(checkpoint_path)
    if ck.get("scene_repr") != "grid":
        raise ValueError(
            "Sugavanam--Ertin stage 1 must be an isotropic grid checkpoint "
            f"(got scene_repr={ck.get('scene_repr')!r})"
        )
    state = ck["model_state_dict"]
    w_re = state["w_re"].detach().cpu()
    w_im = state["w_im"].detach().cpu()
    if w_re.ndim != 3 or w_re.shape != w_im.shape:
        raise ValueError("stage-1 w_re/w_im must be matching [G,G,G] tensors")
    mag = torch.sqrt(w_re.square() + w_im.square())
    active = state.get("active_mask", torch.ones_like(mag, dtype=torch.bool)).detach().cpu().bool()
    pos = state.get("grid_positions")
    if pos is None:
        extent = float(ck["extent"])
        g = int(w_re.shape[0])
        pitch = 2.0 * extent / g
        centres = torch.linspace(-extent + pitch / 2, extent - pitch / 2, g)
        pos = torch.stack(torch.meshgrid(centres, centres, centres, indexing="ij"), dim=-1)
    else:
        pos = pos.detach().cpu()
    vals = mag[active]
    if vals.numel() == 0 or not torch.isfinite(vals).all() or float(vals.max()) <= 0:
        raise ValueError("stage-1 checkpoint contains no finite nonzero scattering centres")
    threshold = threshold_fraction * float(vals.max())
    keep = active & (mag >= threshold)
    flat_idx = torch.nonzero(keep.reshape(-1), as_tuple=False).squeeze(1)
    if flat_idx.numel() < 3:
        raise ValueError(
            f"threshold {threshold_fraction:g} retained only {flat_idx.numel()} points; lower it"
        )
    flat_mag = mag.reshape(-1)
    if max_points > 0 and flat_idx.numel() > max_points:
        _, order = torch.topk(flat_mag[flat_idx], max_points, sorted=False)
        flat_idx = flat_idx[order]
    points = pos.reshape(-1, 3)[flat_idx].numpy().astype(np.float32)
    magnitude = flat_mag[flat_idx].numpy().astype(np.float32)
    extent = float(ck.get("extent") or float(pos.abs().max()))
    granularity = int(ck.get("granularity") or w_re.shape[0])
    pitch = 2.0 * extent / granularity
    radius = float(normal_radius) if normal_radius else 3.0 * pitch
    normals = estimate_pca_normals(points, radius=radius)
    return ScatteringCloud(
        points=points,
        normals=normals,
        magnitude=magnitude,
        threshold=threshold,
        source_epoch=int(ck.get("epoch", -1)),
        source_checkpoint=checkpoint_path,
        extent=extent,
        granularity=granularity,
    )


def spatial_gradient(
    model: nn.Module,
    xyz: torch.Tensor,
    create_graph: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xyz = xyz.requires_grad_(True)
    sdf = model(xyz)
    grad = torch.autograd.grad(
        sdf,
        xyz,
        grad_outputs=torch.ones_like(sdf),
        create_graph=create_graph,
        retain_graph=create_graph,
        only_inputs=True,
    )[0]
    return sdf, grad


def project_to_zero_level(
    model: nn.Module,
    points: torch.Tensor,
    extent: float,
    max_step: float,
    iterations: int = 8,
    tolerance: float = 1e-4,
) -> torch.Tensor:
    """Paper Eqs. 9--10: clipped Newton projection onto f(q)=0."""
    q = points.detach().clone()
    for _ in range(iterations):
        q.requires_grad_(True)
        sdf = model(q)
        grad = torch.autograd.grad(sdf.sum(), q, create_graph=False)[0]
        update = sdf[:, None] * grad / grad.square().sum(-1, keepdim=True).clamp_min(1e-12)
        norm = update.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        update = update * torch.clamp(max_step / norm, max=1.0)
        q = (q - update).detach().clamp(-extent, extent)
        if float(sdf.detach().abs().max()) <= tolerance:
            break
    return q


def uniformize_points(
    points: torch.Tensor,
    bandwidth: float,
    step_size: float = 0.25,
    k: int = 12,
    chunk: int = 1024,
) -> torch.Tensor:
    """Approximate Eq. 11 Gaussian-weighted repulsion in bounded memory."""
    if len(points) < 2:
        return points
    q = points.detach()
    moved = []
    kk = min(k + 1, len(q))
    for start in range(0, len(q), chunk):
        x = q[start : start + chunk]
        dist = torch.cdist(x, q)
        d, idx = torch.topk(dist, kk, largest=False, dim=1)
        d, idx = d[:, 1:], idx[:, 1:]  # exclude self
        neighbours = q[idx]
        delta = x[:, None, :] - neighbours
        direction = delta / d[..., None].clamp_min(1e-12)
        weight = torch.exp(-d.square() / max(bandwidth * bandwidth, 1e-12))
        repel = (weight[..., None] * direction).sum(1) / weight.sum(1, keepdim=True).clamp_min(1e-12)
        moved.append(x + step_size * bandwidth * repel)
    return torch.cat(moved, dim=0)


def refresh_iso_points(
    model: nn.Module,
    seed_points: torch.Tensor,
    extent: float,
    pitch: float,
    n_points: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Projection + uniformization + reprojection from paper Eqs. 9--18.

    The paper's full EAR edge-aware insertion algorithm is not specified well
    enough to reproduce exactly.  We draw/jitter a fixed target count, apply
    its published projection and Gaussian repulsion, and reproject.  This
    approximation is explicit in checkpoint metadata.
    """
    idx = torch.randint(len(seed_points), (n_points,), generator=generator, device=seed_points.device)
    jitter = torch.randn(
        n_points, 3, generator=generator, device=seed_points.device, dtype=seed_points.dtype
    ) * pitch
    q = (seed_points[idx] + jitter).clamp(-extent, extent)
    q = project_to_zero_level(model, q, extent, max_step=2.0 * pitch)
    q = uniformize_points(q, bandwidth=2.0 * pitch).clamp(-extent, extent)
    return project_to_zero_level(model, q, extent, max_step=2.0 * pitch).detach()


def load_sdf_checkpoint(path: str, device: torch.device | str = "cpu") -> Tuple[FourierFeatureSDF, dict]:
    ck = _torch_load(path, map_location=str(device))
    if ck.get("method") != METHOD_NAME:
        raise ValueError(f"not a {METHOD_NAME} checkpoint: {path}")
    model = FourierFeatureSDF(**ck["model_config"]).to(device)
    model.load_state_dict(ck["model_state_dict"])
    return model, ck
