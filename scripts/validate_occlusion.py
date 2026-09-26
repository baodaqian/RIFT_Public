#!/usr/bin/env python
"""Gate for the occlusion term and the --range-model wiring (2026-08-06).

CPU-only, ~30 s. Run before any occlusion arm is queued, and again after any
change to rift/occlusion.py or the geometric gain in either operator.

    python scripts/validate_occlusion.py

Stages
------
A  zeta -> 0 recovers the no-occlusion operator EXACTLY. This is the property
   the whole experimental design leans on: an occlusion arm is a strict
   superset of its baseline, so it can only match or beat it on the training
   objective, and a learned zeta that decays to ~0 is a real negative result
   rather than a broken build.
B  Analytic slab. A uniform extinction field of known sigma over a known
   chord must give exp(-2*sigma*L) to quadrature accuracy. Checks the ray
   march, the box clipping, the half-voxel self-occlusion back-off and the
   1/pitch normalization all at once.
C  Ordering. Two scatterers on one ray: the FRONT one is unattenuated, the
   BACK one is attenuated by the front one's optical depth. This is the
   physics the term exists for.
D  Gauge invariance. w -> a*w must leave T^2 bitwise-comparable, because the
   (gain, scene) scale is an exactly flat direction of the objective and a
   fixed optical depth on a raw |w| would mean nothing.
E  Gradients. Autograd d(T^2)/d(w_re) against central finite differences.
F  DC vs energy keying on a synthetic specular voxel -- the concrete reason
   we do not copy SH-SAS's rho = |sigma_DC|*zeta.
G  --range-model: 'product' and 'sum2' agree between the brute and range
   operators to the fp64 exactness tolerance, and the two MODELS differ by
   <0.3% at this project's 10 m / 2.85 cm operating point (so no current
   result should move) while differing materially at a 0.23 m standoff.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rift.forward_operator import forward_operator_lessparallel, get_kvector
from rift.occlusion import (
    OcclusionScale,
    array_phase_centre,
    opacity_volume,
    ray_transmittance,
    view_transmittance,
)
from rift.range_operator import range_forward_operator
from rift.sparse_scene import SHVoxelGridScene
from rift.config import cc

torch.manual_seed(0)
DEV = torch.device("cpu")
FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def make_scene(g=16, extent=1.5, degree=2, seed=0):
    torch.manual_seed(seed)
    s = SHVoxelGridScene(g, extent, DEV, max_degree=degree, init_degree=degree, init_scale=0.1)
    return s


# ---------------------------------------------------------------- A
print("\nStage A -- zeta -> 0 recovers the no-occlusion operator")
scene = make_scene()
origin = torch.tensor([0.0, 0.0, 10.0])
t2_zero = view_transmittance(scene, OcclusionScale(1e-12), origin, key="energy")
check("T^2 == 1 at zeta -> 0",
      torch.allclose(t2_zero, torch.ones_like(t2_zero), atol=1e-12),
      f"max|T^2-1| = {(t2_zero - 1).abs().max():.3e}")

t2_on = view_transmittance(scene, OcclusionScale(1.0), origin, key="energy")
check("T^2 < 1 somewhere at zeta = 1 (the term actually bites)",
      float(t2_on.min()) < 0.9,
      f"min T^2 = {float(t2_on.min()):.4f}, mean = {float(t2_on.mean()):.4f}")


# ---------------------------------------------------------------- B
print("\nStage B -- analytic uniform slab")
G, EXT = 24, 1.0
pitch = 2.0 * EXT / G
sigma_val = 0.7                                   # 1/m, uniform over the box
sigma_flat = torch.full((G ** 3,), sigma_val)
org = torch.tensor([0.0, 0.0, 5.0])               # outside the box, on +z
# a point on the axis at z = -0.5 : the ray enters the box at z = +1.0
pt = torch.tensor([[0.0, 0.0, -0.5]])
t2 = ray_transmittance(pt, sigma_flat, org, EXT, G)
chord = (1.0 - (-0.5)) - 0.5 * pitch              # entry -> point, minus the back-off
expected = torch.exp(torch.tensor(-2.0 * sigma_val * chord))
check("uniform slab matches exp(-2*sigma*L)",
      bool((t2[0] - expected).abs() < 2e-3),
      f"marched {float(t2[0]):.6f} vs analytic {float(expected):.6f} (chord {chord:.4f} m)")

# a point on the near face should see almost no extinction in front of it
pt_near = torch.tensor([[0.0, 0.0, EXT - 0.5 * pitch]])
t2_near = ray_transmittance(pt_near, sigma_flat, org, EXT, G)
check("near-face point is essentially unoccluded",
      float(t2_near[0]) > 0.99, f"T^2 = {float(t2_near[0]):.6f}")

# same target, different view direction: extinction must be isotropic here
org_x = torch.tensor([5.0, 0.0, 0.0])
pt_x = torch.tensor([[-0.5, 0.0, 0.0]])
t2_x = ray_transmittance(pt_x, sigma_flat, org_x, EXT, G)
check("rotating the view by 90 deg gives the same slab answer",
      bool((t2_x[0] - t2[0]).abs() < 3e-3),
      f"+z {float(t2[0]):.6f} vs +x {float(t2_x[0]):.6f}")


# ---------------------------------------------------------------- C
print("\nStage C -- ordering: the front scatterer shadows the back one")
G, EXT = 16, 1.0
pitch = 2.0 * EXT / G
scene_c = SHVoxelGridScene(G, EXT, DEV, max_degree=0, init_degree=0, init_scale=0.0)
centers = scene_c.grid_positions.reshape(-1, 3)
# an even granularity puts NO voxel centre on the axis, so pick a column by
# index (ix, iy) and aim the ray down it: flat = (ix*G + iy)*G + iz
ix = iy = G // 2
i_front = (ix * G + iy) * G + 12          # z = +0.5625
i_back = (ix * G + iy) * G + 3            # z = -0.5625
with torch.no_grad():
    scene_c.w_re.zero_()
    scene_c.w_im.zero_()
    w = scene_c.w_re.reshape(-1, 1)
    w[i_front, 0] = 1.0
    w[i_back, 0] = 1.0

col_x, col_y = float(centers[i_front, 0]), float(centers[i_front, 1])
org = torch.tensor([col_x, col_y, 8.0])
sigma = opacity_volume(scene_c, OcclusionScale(1.0), key="energy")
t2 = ray_transmittance(scene_c.grid_positions.reshape(-1, 3), sigma, org, EXT, G)
check("front voxel unoccluded", float(t2[i_front]) > 0.999, f"T^2 = {float(t2[i_front]):.6f}")
check("back voxel shadowed by the front one",
      float(t2[i_back]) < 0.2,
      f"T^2 = {float(t2[i_back]):.6f} (expected ~exp(-2*zeta) = {float(torch.exp(torch.tensor(-2.0))):.4f})")
check("the two are ordered correctly", float(t2[i_back]) < float(t2[i_front]))

# and from the OPPOSITE side the roles swap -- the whole point of a per-view term
org_neg = torch.tensor([col_x, col_y, -8.0])
t2_neg = ray_transmittance(scene_c.grid_positions.reshape(-1, 3), sigma, org_neg, EXT, G)
check("viewing from -z swaps which voxel is shadowed",
      float(t2_neg[i_back]) > 0.999 and float(t2_neg[i_front]) < 0.2,
      f"front T^2 {float(t2_neg[i_front]):.4f}, back T^2 {float(t2_neg[i_back]):.4f}")


# ---------------------------------------------------------------- D
print("\nStage D -- gauge invariance of the extinction field")
scene_d = make_scene(g=12, seed=3)
org = torch.tensor([0.0, 3.0, 3.0])
t2_a = view_transmittance(scene_d, OcclusionScale(1.0), org)
with torch.no_grad():
    scene_d.w_re *= 137.0
    scene_d.w_im *= 137.0
t2_b = view_transmittance(scene_d, OcclusionScale(1.0), org)
check("w -> 137*w leaves T^2 unchanged",
      torch.allclose(t2_a, t2_b, rtol=1e-6, atol=1e-8),
      f"max rel diff = {((t2_a - t2_b).abs() / t2_a.abs().clamp_min(1e-12)).max():.3e}")


# ---------------------------------------------------------------- E
print("\nStage E -- autograd vs finite differences")
# everything in fp64: the production march is fp32, in which a central
# difference on one weight is pure rounding noise
scene_e = make_scene(g=8, degree=1, seed=5).double()
org = torch.tensor([0.0, 0.0, 6.0], dtype=torch.float64)
# zeta small enough that the scene is NOT saturated: at zeta 1.5 on a dense
# random scene, every voxel past the front layer sits at T^2 ~ 4e-11 and its
# gradient is numerically zero -- which is exactly why the CLI default starts
# near-transparent and lets training raise the opacity.
scale_e = OcclusionScale(0.1).double()


def loss_fn():
    t2 = view_transmittance(scene_e, scale_e, org, march_dtype=torch.float64)
    return (t2 ** 2).sum()


scene_e.zero_grad(set_to_none=True)
scale_e.zero_grad(set_to_none=True)
loss_fn().backward()
gw = scene_e.w_re.grad.clone()
gz = scale_e.log_zeta.grad.clone()

idx = torch.nonzero(gw.abs().reshape(-1) > 0).reshape(-1)
probe = idx[torch.randperm(idx.numel())[:6]] if idx.numel() else idx
errs = []
h = 1e-4
flatw = scene_e.w_re.data.reshape(-1)
for j in probe.tolist():
    orig = float(flatw[j])
    with torch.no_grad():
        flatw[j] = orig + h
    lp = float(loss_fn())
    with torch.no_grad():
        flatw[j] = orig - h
    lm = float(loss_fn())
    with torch.no_grad():
        flatw[j] = orig
    fd = (lp - lm) / (2 * h)
    an = float(gw.reshape(-1)[j])
    errs.append(abs(fd - an) / max(abs(fd), abs(an), 1e-9))
check("d(loss)/d(w_re) matches finite differences",
      len(errs) > 0 and max(errs) < 2e-3,
      f"max rel err over {len(errs)} probes = {max(errs):.3e}" if errs else "no nonzero grads found")

orig = float(scale_e.log_zeta.data)
with torch.no_grad():
    scale_e.log_zeta.data.fill_(orig + h)
lp = float(loss_fn())
with torch.no_grad():
    scale_e.log_zeta.data.fill_(orig - h)
lm = float(loss_fn())
with torch.no_grad():
    scale_e.log_zeta.data.fill_(orig)
fd = (lp - lm) / (2 * h)
check("d(loss)/d(log_zeta) matches finite differences",
      abs(fd - float(gz)) / max(abs(fd), abs(float(gz)), 1e-9) < 2e-3,
      f"autograd {float(gz):.6e} vs fd {fd:.6e}")

# saturation diagnostic -- the reason the CLI default starts near-transparent
for z in (0.1, 0.5, 1.0, 1.5):
    t2 = view_transmittance(scene_e, OcclusionScale(z).double(), org,
                            march_dtype=torch.float64)
    dead = float((t2 < 1e-6).float().mean())
    print(f"    zeta={z:<4} mean T^2 = {float(t2.mean()):.3e}, "
          f"fraction of voxels below T^2=1e-6 (no usable gradient): {dead:.1%}")


# ---------------------------------------------------------------- F
print("\nStage F -- why opacity is NOT keyed to the DC coefficient")
G, EXT = 8, 1.0
scene_f = SHVoxelGridScene(G, EXT, DEV, max_degree=2, init_degree=2, init_scale=0.0)
with torch.no_grad():
    scene_f.w_re.zero_()
    scene_f.w_im.zero_()
    # a "specular" voxel: no isotropic return, all its energy in l >= 1 --
    # the signature of a flat conducting facet, which is perfectly opaque
    scene_f.w_re.reshape(-1, scene_f.w_re.shape[-1])[10, 0] = 0.0
    scene_f.w_re.reshape(-1, scene_f.w_re.shape[-1])[10, 4] = 1.0
sig_e = opacity_volume(scene_f, OcclusionScale(1.0), key="energy")
sig_d = opacity_volume(scene_f, OcclusionScale(1.0), key="dc")
check("energy keying sees the specular voxel", float(sig_e[10]) > 0)
check("DC keying makes it transparent (SH-SAS's failure mode on PEC)",
      float(sig_d[10]) == 0.0,
      f"sigma_energy = {float(sig_e[10]):.4e}, sigma_dc = {float(sig_d[10]):.4e}")


# ---------------------------------------------------------------- G
print("\nStage G -- --range-model wiring and its size at our operating point")
torch.manual_seed(11)
nf = 64
freqs = torch.linspace(77.5e9, 80.5e9, nf, dtype=torch.float64)
kvec = get_kvector(freqs, cc)
n_el = 4


def array_at(standoff, aperture):
    lin = torch.linspace(-aperture / 2, aperture / 2, n_el, dtype=torch.float64)
    tx = torch.stack([lin, torch.zeros(n_el, dtype=torch.float64),
                      torch.full((n_el,), standoff, dtype=torch.float64)], dim=-1)
    rx = torch.stack([torch.zeros(n_el, dtype=torch.float64), lin,
                      torch.full((n_el,), standoff, dtype=torch.float64)], dim=-1)
    return rx, tx


pos = (torch.rand(64, 3, dtype=torch.float64) - 0.5) * 0.3
w = torch.randn(64, dtype=torch.complex128)

for model in ("sum2", "product", "none"):
    rx, tx = array_at(10.0, 0.0285)
    s_brute = forward_operator_lessparallel(
        freqs, kvec, rx, tx, pos, w, artificial_gain=1.0, p_spectrum=None,
        range_model=model, omega_scaling="unity", phase_sign=-1.0)
    s_range = range_forward_operator(
        freqs, kvec, rx, tx, pos, w, phase_sign=-1.0, compute_dtype=torch.float64,
        range_model=model)
    # brute renders in complex64 by construction; compare at that precision
    rel = ((s_range.to(torch.complex64) - s_brute).abs().sum()
           / s_brute.abs().sum())
    check(f"range vs brute operator agree for range_model={model}",
          float(rel) < 1e-5, f"relative difference {float(rel):.3e}")

# sum2 and product coincide exactly at R_tx = R_rx (they differ by a constant
# factor 4 the global gain absorbs), so what separates them is leg ASYMMETRY,
# i.e. the Tx-Rx separation relative to the range -- and it enters at second
# order. A compact array at 10 m is indistinguishable; a widely separated
# bistatic pair is not.
for standoff, aperture, label in (
        (10.0, 0.0285, "our operating point (10 m standoff, 2.85 cm aperture)"),
        (0.23, 0.20, "wide bistatic (0.23 m standoff, 20 cm Tx-Rx separation)")):
    rx, tx = array_at(standoff, aperture)
    p = (torch.rand(64, 3, dtype=torch.float64) - 0.5) * min(0.3, standoff)
    a = range_forward_operator(freqs, kvec, rx, tx, p, w, phase_sign=-1.0,
                               compute_dtype=torch.float64, range_model="sum2")
    b = range_forward_operator(freqs, kvec, rx, tx, p, w, phase_sign=-1.0,
                               compute_dtype=torch.float64, range_model="product")
    # the models differ by a constant factor 4 in the monostatic limit
    # (1/(2R)^2 vs 1/R^2); compare SHAPES, which is what the gain cannot absorb
    a_n = a / a.abs().max()
    b_n = b / b.abs().max()
    rel = float((a_n - b_n).abs().sum() / a_n.abs().sum())
    print(f"    {label}: shape difference sum2 vs product = {rel:.3%}")
    if standoff == 10.0:
        check("sum2 and product are <0.3% apart at 10 m (no current result moves)",
              rel < 3e-3, f"{rel:.3%}")
    else:
        check("sum2 and product diverge once Tx/Rx are widely separated",
              rel > 3e-3, f"{rel:.3%}")


# ---------------------------------------------------------------- H
print("\nStage H -- array phase centre")
rx, tx = array_at(10.0, 0.0285)
c = array_phase_centre(rx, tx)
check("phase centre sits at the array standoff",
      bool((c - torch.tensor([0.0, 0.0, 10.0], dtype=torch.float64)).abs().max() < 1e-9),
      f"centre = {c.tolist()}")
spread = max(float((tx - c).norm(dim=-1).max()), float((rx - c).norm(dim=-1).max()))
walk = spread * (1.0 / 10.0)      # lateral ray walk 1 m into the scene
print(f"    aperture half-extent {spread * 100:.2f} cm -> ray walk 1 m in = {walk * 1000:.2f} mm "
      f"(g48 @ extent 1.5 pitch = 62.5 mm)")
check("small-aperture approximation holds (walk << voxel pitch)", walk < 0.0625 / 10)


print("\n" + "=" * 62)
if FAILURES:
    print(f"FAILED {len(FAILURES)}: " + ", ".join(FAILURES))
    sys.exit(1)
print("ALL STAGES PASSED")
