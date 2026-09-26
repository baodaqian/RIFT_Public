"""SH-SAS field with CPU-equivalent fp32 coordinate division on PVC.

Probe 2154227 isolated a one-ulp XPU scalar reciprocal error, amplified by
the 4096-resolution hash grid. A same-dtype tensor divisor uses true division
and matches CPU coordinates exactly. All other operations remain unchanged.
"""
import torch

from rift.sh_sas import SHSASField as OriginalSHSASField

BACKEND = "sh_sas_torch_xpu_tensor_divide_v1"


class SHSASField(OriginalSHSASField):
    def query_coefficients(self, points: torch.Tensor, chunk_size: int = 32768) -> torch.Tensor:
        """Copy of the original query; only the XPU extent divisor differs."""
        if points.device.type != "xpu":
            return super().query_coefficients(points, chunk_size)
        if points.ndim != 2 or points.shape[-1] != 3:
            raise ValueError("points must have shape [N,3]")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        outputs = []
        for start in range(0, points.shape[0], chunk_size):
            xyz = points[start : start + chunk_size]
            unit = ((xyz / xyz.new_tensor(self.extent)) + 1.0) * 0.5
            encoded = self.encoder(unit)
            raw = self.mlp(encoded)
            re, im = raw.split(self.n_basis, dim=-1)
            outputs.append(torch.complex(re, im))
        return torch.cat(outputs, dim=0)
