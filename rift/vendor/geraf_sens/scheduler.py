# Copyright (c) 2026 Laboratory of Sensing and Networking Systems, EPFL
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Extracted from commit 38266cb6e194e2f3dcbead614069a7281ffd21a5; see NOTICE.md.
from typing import Dict, List
import math
import torch

class IterLRScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, scheduler_cfgs: List[Dict], total_iters: int):
        self.optimizer = optimizer
        self.scheduler_cfgs = scheduler_cfgs or []
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.total_iters = total_iters

    def step(self, current_iter: int) -> None:
        multipliers = [1.0 for _ in self.base_lrs]
        for cfg in self.scheduler_cfgs:
            if cfg.get("by_epoch", False):
                continue
            begin = int(cfg.get("begin", 0))
            end = cfg.get("end")
            end = self.total_iters if end is None else int(end)
            if current_iter < begin:
                continue
            if current_iter >= end:
                progress = 1.0
            else:
                span = max(end - begin, 1)
                progress = min(max((current_iter - begin) / span, 0.0), 1.0)

            sched_type = cfg.get("type")
            factor = 1.0
            if sched_type == "LinearLR":
                start_factor = float(cfg.get("start_factor", 1.0))
                factor = start_factor + (1.0 - start_factor) * progress
            elif sched_type == "ExponentialLR":
                gamma = cfg.get("gamma")
                if gamma is None:
                    eta_min = float(cfg.get("eta_min", 0.0))
                    for group_idx, base_lr in enumerate(self.base_lrs):
                        if base_lr == 0:
                            continue
                        if eta_min <= 0.0:
                            lr = base_lr if progress == 0.0 else 0.0
                        else:
                            lr = base_lr * ((eta_min / base_lr) ** progress)
                        multipliers[group_idx] *= lr / base_lr
                    continue
                factor = float(gamma) ** max(current_iter - begin, 0)
            elif sched_type == "CosineAnnealingLR":
                eta_min = float(cfg.get("eta_min", 0.0))
                for group_idx, base_lr in enumerate(self.base_lrs):
                    cosine = (1.0 + math.cos(math.pi * progress)) / 2.0
                    lr = eta_min + (base_lr - eta_min) * cosine
                    multipliers[group_idx] *= lr / base_lr if base_lr != 0 else 1.0
                continue
            else:
                continue

            for group_idx in range(len(multipliers)):
                multipliers[group_idx] *= factor

        for group_idx, group in enumerate(self.optimizer.param_groups):
            group["lr"] = self.base_lrs[group_idx] * multipliers[group_idx]
