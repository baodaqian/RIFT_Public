"""Verify the RIFT-PVC environment: pins, imports, and (on a PVC node) XPU checks
including the range operator's own exactness, adjoint and gradient gates."""
import importlib
import importlib.metadata as md
import platform
import sys
from pathlib import Path

import yaml
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
spec = yaml.safe_load((ROOT / "requirements-pvc.yaml").read_text())
assert platform.python_version() == "3.10.16", platform.python_version()
print("Python:", platform.python_version())
# conda-managed build tools: "name=X.Y" in the spec has conda semantics (X.Y.*)
requirements = [f"{d.split('=')[0]}=={d.split('=')[1]}.*" for d in spec["dependencies"]
                if isinstance(d, str) and d.split("=")[0] in ("pip", "setuptools", "wheel")]
requirements += next(d["pip"] for d in spec["dependencies"] if isinstance(d, dict))
for text in requirements:
    if text.startswith("--"):
        continue
    req = Requirement(text)
    version = md.version(req.name)
    assert version in req.specifier, (req.name, version, str(req.specifier))
    print(req.name, version)
for module in ("torch", "numpy", "scipy", "matplotlib", "wandb", "plyfile", "tqdm", "pandas",
               "skimage", "seaborn", "configargparse", "yaml", "imageio", "PIL", "ninja", "rich", "pytest"):
    importlib.import_module(module)
    print("Import OK:", module)
import jaxtyping  # declared RadarSplat dependency
print("Import OK: jaxtyping", md.version("jaxtyping"))
import numpy as np
import open3d as o3d  # conda-forge build (second step); PyPI wheel needs newer glibc
assert md.version("open3d") == "0.19.0", md.version("open3d")
_pc = o3d.geometry.PointCloud(); _pc.points = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]))
assert np.allclose(_pc.get_max_bound(), [1.0, 2.0, 3.0])
print("Import OK: open3d", md.version("open3d"), "(point-cloud bounds OK)")

import torch
assert not torch.cuda.is_available(), "this environment is for XPU; CUDA must not be selected"
x = torch.tensor([1.0, 2.0], requires_grad=True)
x.square().sum().backward()
assert torch.equal(x.grad, torch.tensor([2.0, 4.0]))
xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
print("Torch CPU autograd OK; torch", torch.__version__, "; XPU available:", xpu)
if "--cpu-only" in sys.argv or not xpu:
    print("XPU checks skipped (no XPU device in this process)")
    sys.exit(0)

dev = torch.device("xpu")
props = torch.xpu.get_device_properties(0)
print("XPU:", torch.xpu.get_device_name(0), "| driver", props.driver_version, "| fp64", bool(props.has_fp64),
      "| memory MiB", props.total_memory // 2**20)
assert props.has_fp64, "fp64 unsupported on this device"
a = torch.randn(512, 512, dtype=torch.float64)
assert (a.to(dev) @ a.to(dev) - (a @ a).to(dev)).abs().max().item() < 1e-9, "fp64 matmul"
z = torch.randn(16, 2048, dtype=torch.complex128)
assert ((torch.fft.ifft(z.to(dev), dim=-1).cpu() - torch.fft.ifft(z, dim=-1)).norm() / z.norm()).item() < 1e-14, "complex128 fft"
idx = torch.randint(0, 1024, (4096,)); src = torch.randn(4096, dtype=torch.complex128)
g = torch.zeros(1024, dtype=torch.complex128, device=dev).index_add_(0, idx.to(dev), src.to(dev))
ref = torch.zeros(1024, dtype=torch.complex128).index_add_(0, idx, src)
assert (g.cpu() - ref).abs().max().item() < 1e-12, "complex128 index_add_"
torch.xpu.manual_seed_all(1); torch.xpu.set_rng_state_all(torch.xpu.get_rng_state_all())
p = torch.nn.Parameter(torch.randn(100, dtype=torch.float64, device=dev)); opt = torch.optim.AdamW([p], lr=1e-3)
p.square().sum().backward(); opt.step()
print("XPU fp64 matmul, complex128 FFT and gridding, RNG state, AdamW OK")

# Repo gates on the device: mirror scripts/validate_range_operator.py stages C and D.
sys.path.insert(0, str(ROOT))
from rift.config import cc, spacing
from rift.forward_operator import forward_operator_lessparallel, get_array_pos, get_kvector
from rift.range_operator import range_adjoint_operator, range_forward_operator

def rel_l2(u, v):
    u, v = u.detach().cpu(), v.detach().cpu()
    return float((u - v).norm() / v.norm().clamp_min(1e-30))

torch.manual_seed(2)
freqs = torch.linspace(95e9, 105e9, 16, dtype=torch.float64).to(dev); kvector = get_kvector(freqs, cc)
theta = torch.tensor([[1.0]], dtype=torch.float64, device=dev); phi = torch.tensor([[0.3]], dtype=torch.float64, device=dev)
rx_pos, tx_pos = get_array_pos(theta, phi, 10.0, spacing, 2, 2, dev)
pos0 = (torch.rand(6, 3, dtype=torch.float64) - 0.5).to(dev).requires_grad_()
w_re0 = torch.randn(6, dtype=torch.float64).to(dev).requires_grad_(); w_im0 = torch.randn(6, dtype=torch.float64).to(dev).requires_grad_()
def loss_for_gradcheck(pos, w_re, w_im):
    y = range_forward_operator(freqs, kvector, rx_pos, tx_pos, pos, torch.complex(w_re, w_im), pair_chunk=3, point_chunk=4, compute_dtype=torch.float64)
    return (y.real.square() + y.imag.square()).sum()
assert torch.autograd.gradcheck(loss_for_gradcheck, (pos0, w_re0, w_im0), eps=1e-6, atol=1e-5, rtol=1e-4), "gradcheck"
pos = (torch.rand(6, 3, dtype=torch.float64) - 0.5).to(dev).requires_grad_()
w_re = torch.randn(6, dtype=torch.float64).to(dev).requires_grad_(); w_im = torch.randn(6, dtype=torch.float64).to(dev).requires_grad_()
target = (torch.randn(16, 2, 2, dtype=torch.float64) + 1j * torch.randn(16, 2, 2, dtype=torch.float64)).to(dev)
y_range = range_forward_operator(freqs, kvector, rx_pos, tx_pos, pos, torch.complex(w_re, w_im), pair_chunk=3, point_chunk=4, compute_dtype=torch.float64)
grads_range = torch.autograd.grad((y_range - target).abs().square().sum(), (pos, w_re, w_im))
y_brute = forward_operator_lessparallel(freqs, kvector, rx_pos, tx_pos, pos, torch.complex(w_re, w_im), artificial_gain=1.0, p_spectrum=None,
                                        range_model="sum2", omega_scaling="unity", center_freq_hz=None, phase_sign=1.0)
grads_brute = torch.autograd.grad((y_brute.to(torch.complex128) - target).abs().square().sum(), (pos, w_re, w_im))
rels = [rel_l2(g1, g2) for g1, g2 in zip(grads_range, grads_brute)]
print("Stage C on XPU: gradient match pos/w_re/w_im = " + "/".join(f"{r:.3e}" for r in rels) + " (gate 1e-5)")
assert max(rels) < 1e-5, rels

torch.manual_seed(3)
freqs = torch.linspace(95e9, 105e9, 32, dtype=torch.float64).to(dev); kvector = get_kvector(freqs, cc)
theta = torch.tensor([[1.2]], dtype=torch.float64, device=dev); phi = torch.tensor([[0.4]], dtype=torch.float64, device=dev)
rx_pos, tx_pos = get_array_pos(theta, phi, 10.0, spacing, 4, 4, dev)
pos = ((torch.rand(64, 3, dtype=torch.float64) - 0.5) * 6.0).to(dev)
weights = (torch.randn(64, dtype=torch.float64) + 1j * torch.randn(64, dtype=torch.float64)).to(dev)
y = (torch.randn(32, 4, 4, dtype=torch.float64) + 1j * torch.randn(32, 4, 4, dtype=torch.float64)).to(dev)
ax = range_forward_operator(freqs, kvector, rx_pos, tx_pos, pos, weights, pair_chunk=5, point_chunk=17)
ahy = range_adjoint_operator(freqs, kvector, rx_pos, tx_pos, pos, y, pair_chunk=5, point_chunk=17)
dot_err = float(((ax.conj() * y).sum() - (weights.conj() * ahy).sum()).abs() / (ax.norm() * y.norm()).clamp_min(1e-30))
w_re = weights.real.detach().clone().requires_grad_(); w_im = weights.imag.detach().clone().requires_grad_()
ax2 = range_forward_operator(freqs, kvector, rx_pos, tx_pos, pos, torch.complex(w_re, w_im), pair_chunk=5, point_chunk=17)
grad_re, grad_im = torch.autograd.grad((ax2.conj() * y).sum().real, (w_re, w_im))
vjp_err = rel_l2(torch.complex(grad_re, grad_im), ahy)
print(f"Stage D on XPU: dot test {dot_err:.3e} (gate 1e-10), autograd VJP {vjp_err:.3e} (gate 1e-10)")
assert dot_err < 1e-10 and vjp_err < 1e-10
print("RIFT-PVC environment verification PASSED on", torch.xpu.get_device_name(0))
