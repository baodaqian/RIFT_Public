# Copyright (c) 2026 Laboratory of Sensing and Networking Systems, EPFL
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Extracted from commit 38266cb6e194e2f3dcbead614069a7281ffd21a5; see NOTICE.md.
import torch
import torch.nn as nn
from .field import make_predictor

class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val, activation='exp'):
        super(SingleVarianceNetwork, self).__init__()
        self.act = activation
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        device = x.device
        if self.act=='exp':
            return torch.ones([*x.shape[:-1], 1], device=device) * torch.exp(self.variance * 10.0)
        elif self.act=='linear':
            return torch.ones([*x.shape[:-1], 1], device=device) * self.variance * 10.0
        elif self.act=='square':
            return torch.ones([*x.shape[:-1], 1], device=device) * (self.variance * 10.0) ** 2
        else:
            raise NotImplementedError

    def warp(self, x, inv_s):
        device = x.device
        return torch.ones([*x.shape[:-1], 1], device=device) * inv_s


class ReflectivePowerNetwork(nn.Module):
    default_cfg={
        'human_light': False,
        'sphere_direction': False,
        'light_pos_freq': 8,
        'inner_init': -0.95,
        'roughness_init': 0.0,
        'metallic_init': 0.0,
        'light_exp_max': 0.0,
        'light_act': 'exp',
        'refelective_act': 'sigmoid',
        'light_power': 0.0
    }
    def __init__(self, cfg, feats_dim=256):
        super().__init__()
        self.cfg={**self.default_cfg, **cfg}

        # self.roughness_predictor = make_predictor(feats_dim+3, 1)
        # if self.cfg['roughness_init']!=0:
        #     nn.init.constant_(self.roughness_predictor[-2].bias, self.cfg['roughness_init'])
        self.refelective_predictor = make_predictor(feats_dim + 3, 1, activation=self.cfg['refelective_act'], exp_max=self.cfg['refelective_exp_max'])
        self.register_parameter('light_power', nn.Parameter(torch.tensor(self.cfg['light_power'])))
        # === Initialization to output ~1 ===

    def forward(self, points, feature_vectors, trans_prob):
        # roughness = self.roughness_predictor(torch.cat([feature_vectors, points], -1))
        refelective = self.refelective_predictor(torch.cat([feature_vectors, points], -1))

        color = refelective.squeeze(-1) * torch.exp(self.light_power) * trans_prob
        color = torch.clamp(color, min=1e-6)
        return color
    
    def get_refelectivness(self, points, feature_vectors):
        refelective = self.refelective_predictor(torch.cat([feature_vectors, points], -1))
        return refelective
