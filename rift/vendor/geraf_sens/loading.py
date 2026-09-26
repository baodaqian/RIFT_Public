# Copyright (c) 2026 Laboratory of Sensing and Networking Systems, EPFL
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Extracted from commit 38266cb6e194e2f3dcbead614069a7281ffd21a5; see NOTICE.md.
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from .registry import BaseTransform

def _sample_volume(volume: np.ndarray, points: np.ndarray, mode: str, align_corners: bool) -> np.ndarray:
    normalized_poses = np.stack([points[:, 2], points[:, 1], points[:, 0]], axis=-1)
    grid = torch.tensor(normalized_poses, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    volume_tensor = torch.tensor(volume, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    sampled = F.grid_sample(volume_tensor, grid, mode=mode, align_corners=align_corners)
    return sampled.squeeze().detach().cpu().numpy()


class InterpolateMFAtTargets(BaseTransform):
    def __init__(self, mode: str = "bilinear", align_corners: bool = True) -> None:
        self.sample_mode = mode
        self.align_corners = align_corners

    def transform(self, results: dict) -> Optional[dict]:
        selected_points = results["tgt_sampled_poses_norm"].reshape(-1, 3).copy()
        if "mf_image" in results:
            results["mf_sampled_value"] = _sample_volume(
                results["mf_image"],
                selected_points,
                self.sample_mode,
                self.align_corners,
            )
        if "accmf_image" in results:
            results["accmf_sampled_value"] = _sample_volume(
                results["accmf_image"],
                selected_points,
                self.sample_mode,
                self.align_corners,
            )
        return results
