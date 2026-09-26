"""Implicit neural representation of the radar scattering field.

Maps a (positionally-encoded) 3D voxel position to a complex scattering
coefficient, analogous to a NeRF density/color field but for radar returns.
"""
import math

import torch
import torch.nn as nn
import torch.nn.init as init


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc1_residual = nn.Linear(input_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)

        # Unified head - only 2 channels (real and imaginary parts)
        self.fc6 = nn.Linear(hidden_size, hidden_size)
        self.fc6_residual = nn.Linear(hidden_size, int(hidden_size / 4))
        self.ln6 = nn.LayerNorm(hidden_size)
        self.fc7 = nn.Linear(hidden_size, hidden_size)
        self.ln7 = nn.LayerNorm(hidden_size)
        self.fc8 = nn.Linear(hidden_size, int(hidden_size / 2))
        self.ln8 = nn.LayerNorm(int(hidden_size / 2))
        self.fc9 = nn.Linear(int(hidden_size / 2), int(hidden_size / 4))
        self.fc10 = nn.Linear(int(hidden_size / 4), 2)  # -> 2 scalars per voxel (w_re, w_im)

        self.activation = nn.Tanh()

        self.init_weights()

    def forward(self, x):
        x_deep = self.fc1(x)
        x_res = self.fc1_residual(x)
        x_res = torch.sin(x_res)
        x_deep = self.ln1(x_deep); x_deep = torch.sin(x_deep)
        x_deep = self.fc2(x_deep); x_deep = self.ln2(x_deep); x_deep = torch.sin(x_deep)
        x_deep = self.fc3(x_deep); x_deep = self.ln3(x_deep); x_deep = torch.sin(x_deep)
        x = self.fc4(x_res + x_deep); x = torch.sin(x)
        x_comm = self.fc5(x)

        x_res_part = self.fc6_residual(x_comm); x_res_part = torch.sin(x_res_part)
        x_deep_part = self.fc6(x_comm); x_deep_part = self.ln6(x_deep_part); x_deep_part = torch.sin(x_deep_part)
        x_deep_part = self.fc7(x_deep_part); x_deep_part = self.ln7(x_deep_part); x_deep_part = torch.sin(x_deep_part)
        x_deep_part = self.fc8(x_deep_part); x_deep_part = self.ln8(x_deep_part); x_deep_part = torch.sin(x_deep_part)
        x_deep_part = self.fc9(x_deep_part)

        x_out = self.fc10(x_deep_part + x_res_part)   # [B,2]

        w_re = x_out[:, 0]
        w_im = x_out[:, 1]
        w_complex = torch.complex(w_re, w_im)  # [B], cfloat
        return w_complex

    def init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if name in ['fc1', 'fc1_residual']:
                    n_in = m.in_features; limit = 1.0 / n_in
                    init.uniform_(m.weight, -limit, limit)
                else:
                    n_in = m.in_features; limit = math.sqrt(6.0 / n_in)
                    init.uniform_(m.weight, -limit, limit)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
