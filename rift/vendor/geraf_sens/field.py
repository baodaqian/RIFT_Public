# Copyright (c) 2026 Laboratory of Sensing and Networking Systems, EPFL
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Extracted from commit 38266cb6e194e2f3dcbead614069a7281ffd21a5; see NOTICE.md.
import torch
import torch.nn as nn

class IdentityActivation(nn.Module):
    def forward(self, x): return x


class ExpActivation(nn.Module):
    def __init__(self, max_light=None):
        super().__init__()
        self.max_light=max_light

    def forward(self, x):
        if self.max_light is not None:
            return torch.exp(torch.clamp(x, max=self.max_light))
        else:
            return torch.exp(x)


def make_predictor(feats_dim: object, output_dim: object, weight_norm: object = True, activation='sigmoid', exp_max=None) -> object:
    if activation == 'sigmoid':
        activation = nn.Sigmoid()
    elif activation == 'softplus':
        activation = nn.Softplus()
    elif activation=='exp':
        activation = ExpActivation(max_light=exp_max)
    elif activation=='none':
        activation = IdentityActivation()
    elif activation=='relu':
        activation = nn.ReLU()
    else:
        raise NotImplementedError

    run_dim = 256
    if weight_norm:
        module=nn.Sequential(
            nn.utils.weight_norm(nn.Linear(feats_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, output_dim)),
            activation,
        )
    else:
        module=nn.Sequential(
            nn.Linear(feats_dim, run_dim),
            nn.ReLU(),
            nn.Linear(run_dim, run_dim),
            nn.ReLU(),
            nn.Linear(run_dim, run_dim),
            nn.ReLU(),
            nn.Linear(run_dim, output_dim),
            activation,
        )

    return module
