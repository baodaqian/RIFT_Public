"""Radar-only Radar Fields baseline for the RIFT measurement geometry.

The behavioral reference is the official SIGGRAPH 2024 release pinned in
``external/RADAR_FIELDS_REFERENCE.md``.  This module intentionally contains no
auxiliary-sensor data path.  It keeps the released decomposition

    radar cross section = occupancy(x) * reflectance(x, view_direction)

and its log-intensity reconstruction loss, while replacing the original
mechanically scanned azimuth/range cells with differentiable bistatic range
cells constructed from RIFT's measured Tx/Rx positions.

This module preserves the historical PyTorch model and voxel renderer and
provides a separately identified portable audited encoder. The audited default
uses the original TCNN model in ``radar_fields_upstream`` and bin-centered ray
integration in ``radar_fields_native``; the portable model is not an exact
replacement for the original fused networks and direction convention.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rift.spherical_harmonics import real_sh_basis


OFFICIAL_REFERENCE_COMMIT = "ee76d76570f58b3d8539eafd7df0c188b58af333"
NORMALIZED_DB_INTENSITY_DOMAIN = "normalized_dB_range_power_intensity"
NORMALIZED_DB_INTENSITY_LABEL = "normalized-dB range-power intensity"
NORMALIZED_DB_RELMSE_LABEL = "relative MSE in normalized-dB range-power intensity"


class HashGridEncoder(nn.Module):
    """Versioned multi-resolution trilinear hash encoding.

    Defaults are the official configuration: 16 levels, two features per
    level, base resolution 16, final resolution 512 and a 2**19 entry cap.
    Inputs must already be normalized to the unit cube. ``legacy`` preserves
    historical indexing; ``tcnn`` uses dense/hashed vertex-grid conventions.
    """

    _PRIMES = (1, 2654435761, 805459861)

    def __init__(
        self,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        base_resolution: int = 16,
        final_resolution: int = 512,
        log2_hashmap_size: int = 19,
        layout: str = "legacy",
    ) -> None:
        super().__init__()
        if n_levels < 1:
            raise ValueError("n_levels must be positive")
        if base_resolution < 2 or final_resolution < base_resolution:
            raise ValueError("invalid hash-grid resolution range")
        if layout not in ("legacy", "tcnn"):
            raise ValueError("hash layout must be legacy or tcnn")
        self.layout = layout

        self.n_levels = int(n_levels)
        self.n_features_per_level = int(n_features_per_level)
        self.base_resolution = int(base_resolution)
        self.final_resolution = int(final_resolution)
        self.max_entries = 1 << int(log2_hashmap_size)
        if self.n_levels == 1:
            scale = 1.0
        else:
            scale = math.exp(
                math.log(self.final_resolution / self.base_resolution)
                / (self.n_levels - 1)
            )
        resolutions = [
            max(2, int(math.floor(self.base_resolution * (scale ** level))))
            for level in range(self.n_levels)
        ]
        resolutions[-1] = self.final_resolution
        self.scales = tuple(self.base_resolution * scale ** level - 1.0
                            for level in range(self.n_levels))
        if layout == "tcnn":
            # TCNN counts vertices, staggers coordinates by 0.5, pads tables
            # to eight entries, and hashes only levels that exceed capacity.
            self.scales = (*self.scales[:-1], float(self.final_resolution - 1))
            resolutions = [int(math.ceil(value)) + 1 for value in self.scales]
        self.resolutions = tuple(resolutions)

        tables = []
        for resolution in self.resolutions:
            entries = min(self.max_entries, (resolution + 1) ** 3)
            if layout == "tcnn":
                entries = min(self.max_entries, ((resolution ** 3 + 7) // 8) * 8)
            embedding = nn.Embedding(entries, self.n_features_per_level)
            nn.init.uniform_(embedding.weight, -1.0e-4, 1.0e-4)
            tables.append(embedding)
        self.tables = nn.ModuleList(tables)

        corners = torch.tensor(
            [[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)],
            dtype=torch.long,
        )
        self.register_buffer("corners", corners, persistent=False)

    @property
    def output_dim(self) -> int:
        return self.n_levels * self.n_features_per_level

    @classmethod
    def _hash(cls, coords: torch.Tensor, table_size: int) -> torch.Tensor:
        coords = coords.to(torch.int64)
        hashed = (
            coords[..., 0] * cls._PRIMES[0]
            ^ coords[..., 1] * cls._PRIMES[1]
            ^ coords[..., 2] * cls._PRIMES[2]
        )
        return torch.remainder(hashed, table_size).long()

    def forward(self, unit_xyz: torch.Tensor, mask_progress: Optional[float] = None) -> torch.Tensor:
        if unit_xyz.ndim != 2 or unit_xyz.shape[-1] != 3:
            raise ValueError(f"unit_xyz must have shape [N,3], got {tuple(unit_xyz.shape)}")
        if self.layout == "tcnn":
            if not torch.isfinite(unit_xyz).all() or ((unit_xyz < 0) | (unit_xyz > 1)).any():
                raise ValueError("audited hash queries must lie in the finite unit cube")
            xyz = unit_xyz
        else:
            xyz = unit_xyz.clamp(0.0, 1.0 - 1.0e-7)
        encoded = []
        corners = self.corners.to(xyz.device)

        for level, (resolution, table) in enumerate(zip(self.resolutions, self.tables)):
            scaled = (xyz * self.scales[level] + 0.5 if self.layout == "tcnn"
                      else xyz * float(resolution))
            base = torch.floor(scaled).long()
            frac = scaled - base.to(scaled.dtype)
            corner_coords = base[:, None, :] + corners[None, :, :]
            if self.layout == "tcnn" and resolution ** 3 <= table.num_embeddings:
                indices = (corner_coords[..., 0] + resolution * corner_coords[..., 1]
                           + resolution ** 2 * corner_coords[..., 2]) % table.num_embeddings
            else:
                indices = self._hash(corner_coords, table.num_embeddings)
            features = table(indices)  # [N,8,F]
            corner_f = corners.to(frac.dtype)[None, :, :]
            weights = torch.where(
                corner_f.bool(), frac[:, None, :], 1.0 - frac[:, None, :]
            ).prod(dim=-1)
            encoded.append((features * weights[..., None]).sum(dim=1))

        out = torch.cat(encoded, dim=-1)
        if mask_progress is not None:
            progress = float(max(0.0, min(1.0, mask_progress)))
            visible_fraction = 0.4 + 0.6 * progress
            visible = int(math.ceil(visible_fraction * out.shape[-1]))
            if visible < out.shape[-1]:
                mask = torch.arange(out.shape[-1], device=out.device) < visible
                out = out * mask.to(out.dtype)
        return out


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, hidden_layers: int = 1, *, bias: bool = True) -> nn.Sequential:
    layers = []
    current = in_dim
    for _ in range(hidden_layers):
        layers.extend((nn.Linear(current, hidden_dim, bias=bias), nn.ReLU()))
        current = hidden_dim
    layers.append(nn.Linear(current, out_dim, bias=bias))
    return nn.Sequential(*layers)


class RadarFieldsModel(nn.Module):
    """Occupancy and view-conditioned reflectance fields.

    The split architecture follows the released ``RadarField`` model: a shared
    32-D spatial feature, one sigmoid occupancy head, and one softplus
    reflectance head conditioned on a degree-3 real SH encoding (16 values,
    corresponding to tiny-cuda-nn's ``SphericalHarmonics degree=4`` output).
    """

    def __init__(
        self,
        extent: float,
        hidden_dim: int = 64,
        feature_dim: int = 32,
        sh_degree: int = 3,
        sigmoid_tightness: float = 1.0,
        batch_norm: bool = True,
        hash_levels: int = 16,
        hash_features: int = 2,
        hash_base_resolution: int = 16,
        hash_final_resolution: int = 512,
        hash_log2_size: int = 19,
        encoding_layout: str = "legacy",
    ) -> None:
        super().__init__()
        if extent <= 0:
            raise ValueError("extent must be positive")
        self.extent = float(extent)
        self.sh_degree = int(sh_degree)
        self.sigmoid_tightness = float(sigmoid_tightness)
        self.encoding_layout = encoding_layout

        self.xyz_encoding = HashGridEncoder(
            n_levels=hash_levels,
            n_features_per_level=hash_features,
            base_resolution=hash_base_resolution,
            final_resolution=hash_final_resolution,
            log2_hashmap_size=hash_log2_size,
            layout=encoding_layout,
        )
        bias = encoding_layout == "legacy"  # FullyFusedMLP has no bias parameters.
        self.xyz_net = _mlp(self.xyz_encoding.output_dim, hidden_dim, feature_dim, hidden_layers=1, bias=bias)
        self.feature_norm = nn.BatchNorm1d(feature_dim) if batch_norm else nn.Identity()
        self.alpha_net = _mlp(feature_dim, hidden_dim, 1, hidden_layers=1, bias=bias)
        self.reflectance_net = _mlp(
            feature_dim + (self.sh_degree + 1) ** 2,
            hidden_dim,
            1,
            hidden_layers=1,
            bias=bias,
        )

    def normalize_xyz(self, xyz: torch.Tensor) -> torch.Tensor:
        return (xyz + self.extent) / (2.0 * self.extent)

    def encode_direction(self, direction: torch.Tensor) -> torch.Tensor:
        if direction.shape[-1] != 3 or direction.ndim not in (1, 2):
            raise ValueError(f"direction must have shape [3] or [N,3], got {tuple(direction.shape)}")
        direction = direction / torch.linalg.vector_norm(
            direction, dim=-1, keepdim=True
        ).clamp_min(1.0e-12)
        theta = torch.acos(direction[..., 2].clamp(-1.0, 1.0))
        phi = torch.atan2(direction[..., 1], direction[..., 0])
        # real_sh_basis is basis-major for vector inputs; the MLP expects
        # the feature axis last, matching tiny-cuda-nn's encoding contract.
        return real_sh_basis(theta, phi, self.sh_degree).movedim(0, -1).to(direction.dtype)

    def forward(
        self,
        xyz: torch.Tensor,
        view_direction: torch.Tensor,
        mask_progress: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        encoded = self.xyz_encoding(self.normalize_xyz(xyz), mask_progress=mask_progress)
        features = self.feature_norm(self.xyz_net(encoded))
        return self._heads(features, view_direction)

    def _heads(self, features, view_direction):
        alpha = torch.sigmoid(self.alpha_net(features) * self.sigmoid_tightness).squeeze(-1)
        direction_features = self.encode_direction(view_direction)
        if direction_features.ndim == 1:
            direction_features = direction_features[None, :].expand(features.shape[0], -1)
        elif direction_features.shape[0] != features.shape[0]:
            raise ValueError("per-point view directions must match xyz")
        inputs = ((direction_features, features) if self.encoding_layout == "tcnn"
                  else (features, direction_features))
        reflectance = F.softplus(self.reflectance_net(torch.cat(inputs, dim=-1))).squeeze(-1)
        return {"alpha": alpha, "reflectance": reflectance, "rcs": alpha * reflectance}

    def query_chunked(
        self,
        xyz: torch.Tensor,
        view_direction: torch.Tensor,
        mask_progress: Optional[float] = None,
        chunk_size: int = 32768,
    ) -> Dict[str, torch.Tensor]:
        if chunk_size <= 0 or xyz.shape[0] == 0:
            raise ValueError("queries and chunk_size must be nonempty/positive")
        if self.encoding_layout == "tcnn" and self.training and isinstance(self.feature_norm, nn.BatchNorm1d):
            # A memory chunk is not a BN minibatch. Compute moments once for
            # all queried samples; changing chunk_size must not change the
            # statistical model or update running moments multiple times.
            features = torch.cat([
                self.xyz_net(self.xyz_encoding(self.normalize_xyz(part), mask_progress))
                for part in xyz.split(chunk_size)
            ])
            if features.shape[0] == 1:
                features = F.batch_norm(features, self.feature_norm.running_mean,
                                        self.feature_norm.running_var, self.feature_norm.weight,
                                        self.feature_norm.bias, training=False, eps=self.feature_norm.eps)
            else:
                features = self.feature_norm(features)
            return self._heads(features, view_direction)
        outputs = {"alpha": [], "reflectance": [], "rcs": []}
        for start in range(0, xyz.shape[0], chunk_size):
            stop = start + chunk_size
            chunk_direction = (
                view_direction[start:stop] if view_direction.ndim == 2 else view_direction
            )
            out = self(xyz[start:stop], chunk_direction, mask_progress)
            for key in outputs:
                outputs[key].append(out[key])
        return {key: torch.cat(value, dim=0) for key, value in outputs.items()}


def radar_fields_intensity(
    rcs: torch.Tensor,
    ranges: torch.Tensor,
    offset: float = 0.05,
    scaler: float = 1.0,
    range_law: str = "released",
) -> torch.Tensor:
    """Map nonnegative RCS to the released log-intensity domain.

    ``released`` is the official config's default ``approx=True`` path.
    ``code_r2`` reproduces the release's non-approximate branch. ``paper_r4``
    exposes Eq. 9 of the paper instead of silently resolving that discrepancy.
    """

    if range_law == "released":
        adjusted = rcs
    elif range_law == "code_r2":
        adjusted = rcs / ranges.clamp_min(1.0e-9).square()
    elif range_law == "paper_r4":
        adjusted = rcs / ranges.clamp_min(1.0e-9).pow(4)
    else:
        raise ValueError(f"unknown range_law {range_law!r}")
    return torch.log10(adjusted.clamp_min(0.0) + float(offset)) * float(scaler)


def _pair_elements(pair_indices: torch.Tensor, num_rx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.div(pair_indices, num_rx, rounding_mode="floor"), torch.remainder(pair_indices, num_rx)


def bistatic_range_cells(
    values: torch.Tensor,
    xyz: torch.Tensor,
    tx_pos: torch.Tensor,
    rx_pos: torch.Tensor,
    pair_indices: torch.Tensor,
    bin_size: float,
    num_bins: int,
    pair_chunk: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Average voxel values into differentiable bistatic range cells.

    ``values`` is ``[N,C]`` (for example occupancy and RCS).  Fractional
    one-way range ``(||x-tx||+||x-rx||)/2`` is linearly splatted into adjacent
    bins.  The return is ``[P,num_bins,C]`` plus the interpolation mass
    ``[P,num_bins]``. Averaging, rather than coherent summation, matches Radar
    Fields' interpretation of each beam/range cell as a sampled local RCS.
    """

    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] != xyz.shape[0]:
        raise ValueError("values must have shape [N] or [N,C] matching xyz")
    if pair_indices.ndim != 1:
        raise ValueError("pair_indices must be one-dimensional")
    if bin_size <= 0 or num_bins <= 0:
        raise ValueError("bin_size and num_bins must be positive")

    num_rx = rx_pos.shape[0]
    tx_idx, rx_idx = _pair_elements(pair_indices.long(), num_rx)
    chunks = []
    mass_chunks = []
    channels = values.shape[1]

    for start in range(0, pair_indices.numel(), pair_chunk):
        stop = min(start + pair_chunk, pair_indices.numel())
        tx = tx_pos[tx_idx[start:stop]]
        rx = rx_pos[rx_idx[start:stop]]
        p = tx.shape[0]
        d_tx = torch.linalg.vector_norm(xyz[None, :, :] - tx[:, None, :], dim=-1)
        d_rx = torch.linalg.vector_norm(xyz[None, :, :] - rx[:, None, :], dim=-1)
        bin_position = (0.5 * (d_tx + d_rx)) / float(bin_size)
        lower = torch.floor(bin_position).long()
        fraction = bin_position - lower.to(bin_position.dtype)

        out = torch.zeros((p * num_bins, channels), dtype=values.dtype, device=values.device)
        mass = torch.zeros((p * num_bins,), dtype=values.dtype, device=values.device)
        pair_offset = torch.arange(p, device=values.device)[:, None] * num_bins
        expanded_values = values[None, :, :].expand(p, -1, -1)

        for indices, weights in ((lower, 1.0 - fraction), (lower + 1, fraction)):
            valid = (indices >= 0) & (indices < num_bins)
            flat_indices = (pair_offset + indices.clamp(0, num_bins - 1))[valid]
            source = (expanded_values * weights[..., None])[valid]
            out = out.index_add(0, flat_indices, source)
            mass = mass.index_add(0, flat_indices, weights[valid].to(values.dtype))

        out = out.reshape(p, num_bins, channels)
        mass = mass.reshape(p, num_bins)
        chunks.append(out / mass.clamp_min(1.0e-12)[..., None])
        mass_chunks.append(mass)

    return torch.cat(chunks, dim=0), torch.cat(mass_chunks, dim=0)


def radar_fields_loss(
    pred_intensity: torch.Tensor,
    target_intensity: torch.Tensor,
    pred_occupancy: torch.Tensor,
    target_occupancy: torch.Tensor,
    weight_fft: float = 0.60,
    weight_occ: float = 0.36,
    weight_bimodal: float = 0.03,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Released reconstruction, occupancy-KL and bimodality objectives."""

    fft_loss = F.l1_loss(pred_intensity, target_intensity)

    pred_flat = pred_occupancy.clamp_min(1.0e-12).reshape(-1)
    target_flat = target_occupancy.clamp_min(0.0).reshape(-1)
    pred_dist = pred_flat / pred_flat.sum().clamp_min(1.0e-12)
    target_dist = target_flat / target_flat.sum().clamp_min(1.0e-12)
    occ_loss = F.kl_div(pred_dist.log(), target_dist, reduction="sum")

    occupied = target_occupancy > 0.5
    bimodal = pred_occupancy.new_zeros(())
    if occupied.any():
        bimodal = bimodal + pred_occupancy[occupied].float().std(unbiased=False)
    if (~occupied).any():
        bimodal = bimodal + pred_occupancy[~occupied].float().std(unbiased=False)

    terms = {
        "fft": fft_loss * float(weight_fft),
        "occupancy": occ_loss * float(weight_occ),
        "bimodal": bimodal * float(weight_bimodal),
    }
    return sum(terms.values()), terms


def padded_roi_objective_diagnostics(
    pred_intensity: torch.Tensor,
    target_intensity: torch.Tensor,
    valid_cells: torch.Tensor,
    parameters: Iterable[torch.nn.Parameter],
    *,
    weight_fft: float,
) -> Dict[str, float]:
    """Measure, but do not alter, padded-cell FFT-loss behavior.

    ``view_objective`` currently zeroes the rendered occupancy/RCS outside
    cells hit by the scene grid, while retaining the corresponding target
    intensities in the released FFT L1 objective.  This helper reports the
    exact valid and padded partitions of that *existing* mean L1 term.  It
    additionally evaluates their separate parameter-gradient L2 norms without
    populating ``.grad``.  It is deliberately diagnostic-only: it neither
    masks a target nor changes the loss used by legacy recipes.

    The reported partition contributions are normalized by the full ROI cell
    count, so ``valid_fft_l1_full_mean + padded_fft_l1_full_mean`` equals
    ``torch.nn.functional.l1_loss(pred_intensity, target_intensity)`` up to
    floating-point reduction order.  Gradient norms use those same
    full-mean contributions, before ``weight_fft`` is applied.
    """

    if pred_intensity.shape != target_intensity.shape:
        raise ValueError("pred_intensity and target_intensity must have matching shapes")
    if valid_cells.shape != pred_intensity.shape:
        raise ValueError("valid_cells must have the same shape as the intensity tensors")
    if pred_intensity.numel() == 0:
        raise ValueError("ROI diagnostic requires at least one cell")

    valid = valid_cells.to(torch.bool)
    padded = ~valid
    total_count = int(valid.numel())
    valid_count = int(valid.sum().item())
    padded_count = total_count - valid_count
    absolute_error = (pred_intensity - target_intensity).abs()

    def contribution(mask: torch.Tensor) -> torch.Tensor:
        # Dividing both partitions by the full count makes them additive parts
        # of the current F.l1_loss reduction rather than conditional means.
        return absolute_error[mask].sum() / float(total_count)

    valid_contribution = contribution(valid)
    padded_contribution = contribution(padded)
    trainable = tuple(parameter for parameter in parameters if parameter.requires_grad)

    def gradient_l2(partition_loss: torch.Tensor) -> float:
        if not trainable or not partition_loss.requires_grad:
            return 0.0
        gradients = torch.autograd.grad(
            partition_loss,
            trainable,
            retain_graph=True,
            allow_unused=True,
        )
        squared_norm = sum(
            (gradient.detach().square().sum() for gradient in gradients if gradient is not None),
            start=partition_loss.detach().new_zeros(()),
        )
        return float(torch.sqrt(squared_norm).cpu())

    valid_gradient_l2 = gradient_l2(valid_contribution)
    padded_gradient_l2 = gradient_l2(padded_contribution)
    return {
        "roi_cell_count": float(total_count),
        "valid_roi_cell_count": float(valid_count),
        "padded_roi_cell_count": float(padded_count),
        "padded_roi_fraction": float(padded_count / total_count),
        "valid_fft_l1_full_mean": float(valid_contribution.detach()),
        "padded_fft_l1_full_mean": float(padded_contribution.detach()),
        "fft_l1_full_mean": float((valid_contribution + padded_contribution).detach()),
        "valid_fft_weighted_objective": float((valid_contribution * weight_fft).detach()),
        "padded_fft_weighted_objective": float((padded_contribution * weight_fft).detach()),
        "valid_fft_model_gradient_l2": valid_gradient_l2,
        "padded_fft_model_gradient_l2": padded_gradient_l2,
    }


def intensity_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    error = pred - target
    sq_error = error.square().sum().item()
    target_power = target.square().sum().item()
    count = target.numel()
    mse = sq_error / max(1, count)
    return {
        "rel_mse": sq_error / max(target_power, 1.0e-30),
        "rmse": math.sqrt(mse),
        "psnr_db": -10.0 * math.log10(max(mse, 1.0e-30)),
        "sq_error": sq_error,
        "target_power": target_power,
        "count": float(count),
    }
