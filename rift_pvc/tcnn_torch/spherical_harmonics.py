"""SphericalHarmonics encoding: torch mirror of tiny-cuda-nn ``kernel_sh``.

Transcribed from the pinned ``include/tiny-cuda-nn/encodings/spherical_harmonics.h``
(inputs mapped ``data_in * 2.f - 1.f``, ``degree * degree`` outputs, no
parameters) and ``common_device.h`` ``sh_enc`` (the real-SH polynomial table
with tiny-cuda-nn's coefficient signs, generated from the recurrences in
appendix A1 of Sloan's "Stupid SH tricks"). Values are evaluated in float32
and cast to the output dtype entry by entry, as the kernel casts each
``(T)(...)``. The input gradient is autograd's, equal to the kernel's
``2 * sh_enc_grad``.

The table is transcribed up to degree 4 (16 coefficients, the release's
``degree`` for ``angle_encoding="SphericalHarmonics"``); degrees 5-8 exist in
the source but are not mirrored and raise ``NotImplementedError``.

A ``tcnn.Module`` with no parameters still registers an empty ``params``
Parameter (``initial_params`` returns a zero-length tensor); the authors'
``get_params`` therefore yields an empty parameter group for ``encode_angle``,
and so does this shim.
"""
from __future__ import annotations

import torch

from .module import DEFAULT_SEED, ShimModule, resolve_dtype, scale_grad

MAX_DEGREE = 4


def sh_enc(degree: int, d: torch.Tensor) -> torch.Tensor:
    """``sh_enc`` for directions ``d`` ``[N, 3]`` (already mapped to ``2x-1``)."""
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    xy, xz, yz, x2, y2, z2 = x * y, x * z, y * z, x * x, y * y, z * z
    cols = [torch.full_like(x, 0.28209479177387814)]                       # 1/(2*sqrt(pi))
    if degree > 1:
        cols += [-0.48860251190291987 * y,                                  # -sqrt(3)*y/(2*sqrt(pi))
                 0.48860251190291987 * z,                                   # sqrt(3)*z/(2*sqrt(pi))
                 -0.48860251190291987 * x]                                  # -sqrt(3)*x/(2*sqrt(pi))
    if degree > 2:
        cols += [1.0925484305920792 * xy,                                   # sqrt(15)*xy/(2*sqrt(pi))
                 -1.0925484305920792 * yz,                                  # -sqrt(15)*yz/(2*sqrt(pi))
                 0.94617469575755997 * z2 - 0.31539156525251999,            # sqrt(5)*(3*z2 - 1)/(4*sqrt(pi))
                 -1.0925484305920792 * xz,                                  # -sqrt(15)*xz/(2*sqrt(pi))
                 0.54627421529603959 * x2 - 0.54627421529603959 * y2]       # sqrt(15)*(x2 - y2)/(4*sqrt(pi))
    if degree > 3:
        cols += [0.59004358992664352 * y * (-3.0 * x2 + y2),                # sqrt(70)*y*(-3*x2 + y2)/(8*sqrt(pi))
                 2.8906114426405538 * xy * z,                               # sqrt(105)*xy*z/(2*sqrt(pi))
                 0.45704579946446572 * y * (1.0 - 5.0 * z2),                # sqrt(42)*y*(1 - 5*z2)/(8*sqrt(pi))
                 0.3731763325901154 * z * (5.0 * z2 - 3.0),                 # sqrt(7)*z*(5*z2 - 3)/(4*sqrt(pi))
                 0.45704579946446572 * x * (1.0 - 5.0 * z2),                # sqrt(42)*x*(1 - 5*z2)/(8*sqrt(pi))
                 1.4453057213202769 * z * (x2 - y2),                        # sqrt(105)*z*(x2 - y2)/(4*sqrt(pi))
                 0.59004358992664352 * x * (-x2 + 3.0 * y2)]                # sqrt(70)*x*(-x2 + 3*y2)/(8*sqrt(pi))
    return torch.stack(cols, dim=-1)


class SphericalHarmonicsEncoding(ShimModule):
    def __init__(self, n_input_dims: int, encoding_config: dict, seed: int = DEFAULT_SEED, dtype=None):
        cfg = dict(encoding_config)
        degree = int(cfg.get("degree", 4))    # encoding.cu: encoding.value("degree", 4u)
        if n_input_dims != 3:
            raise RuntimeError("Can only encode 3D directions in spherical harmonics.")
        if degree <= 0:
            raise RuntimeError("Spherical harmonics must have positive degree.")
        if degree > 8:
            raise RuntimeError("Spherical harmonics are only implemented up to degree 8.")
        if degree > MAX_DEGREE:
            raise NotImplementedError(f"torch shim transcribes sh_enc up to degree {MAX_DEGREE}; got {degree}")
        super().__init__(3, degree * degree, seed, resolve_dtype(dtype))
        self.encoding_config = cfg
        self.degree = degree
        self._set_params(torch.zeros(0, dtype=torch.float32))

    def hyperparams(self) -> dict:
        return {"otype": "SphericalHarmonics", "degree": self.degree}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._check_input(x)
        if self.dtype == torch.float16:
            x = scale_grad(x, 1.0 / self.loss_scale)
            return scale_grad(sh_enc(self.degree, x * 2.0 - 1.0).to(self.dtype), self.loss_scale)
        return sh_enc(self.degree, x * 2.0 - 1.0).to(self.dtype)
