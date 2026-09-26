#!/usr/bin/env python
"""Validate rift/range_operator.py.

The range operator takes the full uniform frequency grid and optionally
gathers a selected subset after rendering.  This script checks exactness
against direct complex128 sums, gradients, adjoints, and the real float32
frequency path used by CSVSimulationDataset.
"""

import argparse
import glob
import math
import os
import random
import sys
import time

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rift.config import cc, extent, spacing
from rift.dataset import CSVSimulationDataset
from rift.encoding import generate_dynamic_grid
from rift.forward_operator import (
    forward_operator_lessparallel,
    get_array_pos,
    get_kvector,
)
from rift.range_operator import range_adjoint_operator, range_forward_operator


REAL_SPHERE_CSV_DIR = "/storage/project/r-jromberg3-0/dbao31/Radar_Opt/ansysData/AEDT_Sphere_Repeat_CSV"
EPS = 1e-30


def direct_forward(freqs, kvector, rx_pos, tx_pos, scatterer_pos, scatterer_weights, phase_sign=1.0, eps=1e-9):
    freqs = freqs.to(dtype=torch.float64, device=scatterer_pos.device)
    kvector = kvector.to(dtype=torch.float64, device=scatterer_pos.device)
    rx_pos = rx_pos.to(dtype=torch.float64, device=scatterer_pos.device)
    tx_pos = tx_pos.to(dtype=torch.float64, device=scatterer_pos.device)
    pos = scatterer_pos.to(dtype=torch.float64)
    w = scatterer_weights.to(torch.complex128)

    r_tx = torch.linalg.norm(pos[:, None, :] - tx_pos[None, :, :], dim=-1).clamp_min(eps)
    r_rx = torch.linalg.norm(pos[:, None, :] - rx_pos[None, :, :], dim=-1).clamp_min(eps)
    r_sum = r_tx[:, :, None] + r_rx[:, None, :]
    geom = 1.0 / (r_sum.square() + eps)
    g_const = 1.0 / ((4.0 * torch.pi) ** 2)

    out = []
    for k_i in kvector:
        field = w[:, None, None] * g_const * geom * torch.exp(1j * phase_sign * k_i * r_sum)
        out.append(field.sum(dim=0).transpose(0, 1))
    return torch.stack(out, dim=0)


def direct_adjoint(freqs, kvector, rx_pos, tx_pos, scatterer_pos, s_resid, phase_sign=1.0, eps=1e-9):
    kvector = kvector.to(dtype=torch.float64, device=scatterer_pos.device)
    rx_pos = rx_pos.to(dtype=torch.float64, device=scatterer_pos.device)
    tx_pos = tx_pos.to(dtype=torch.float64, device=scatterer_pos.device)
    pos = scatterer_pos.to(dtype=torch.float64)
    resid = s_resid.to(torch.complex128)

    r_tx = torch.linalg.norm(pos[:, None, :] - tx_pos[None, :, :], dim=-1).clamp_min(eps)
    r_rx = torch.linalg.norm(pos[:, None, :] - rx_pos[None, :, :], dim=-1).clamp_min(eps)
    r_sum = r_tx[:, :, None] + r_rx[:, None, :]
    geom = 1.0 / (r_sum.square() + eps)
    g_const = 1.0 / ((4.0 * torch.pi) ** 2)

    out = torch.zeros(pos.shape[0], dtype=torch.complex128, device=pos.device)
    for i, k_i in enumerate(kvector):
        ker = g_const * geom * torch.exp(-1j * phase_sign * k_i * r_sum)
        out += torch.einsum("ntr,rt->n", ker, resid[i])
    return out


def rel_l2(a, b):
    return float((a - b).norm() / b.norm().clamp_min(EPS))


def global_rel_mse(pred, gt):
    return float((pred - gt).abs().square().sum() / gt.abs().square().sum().clamp_min(EPS))


def random_viewpoints(n, seed, device):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    theta = torch.pi / 2 + (torch.rand(n, generator=gen) - 0.5) * torch.pi * 0.6
    phi = torch.rand(n, generator=gen) * 2 * torch.pi
    return theta.to(device), phi.to(device)


def real_sphere_viewpoints(n, directory, seed, device):
    files = sorted(glob.glob(os.path.join(directory, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {directory}")
    random.seed(seed)
    chosen = random.sample(files, min(n, len(files)))
    thetas, phis = [], []
    for f in chosen:
        base = os.path.basename(f).replace(".csv", "")
        tokens = base.split("_")
        phis.append(float(tokens[tokens.index("dphi") + 1]))
        thetas.append(float(tokens[tokens.index("dtheta") + 1]))
    return torch.tensor(thetas, device=device), torch.tensor(phis, device=device)


def stage_a(args):
    print("\nStage A: micro exactness (CPU, fp64)")
    device = torch.device("cpu")
    torch.manual_seed(args.seed)
    freqs = torch.linspace(95e9, 105e9, 32, dtype=torch.float64, device=device)
    kvector = get_kvector(freqs, cc)
    pos = (torch.rand(64, 3, dtype=torch.float64, device=device) - 0.5) * 6.0
    weights = torch.randn(64, dtype=torch.float64, device=device) + 1j * torch.randn(64, dtype=torch.float64, device=device)
    thetas, phis = random_viewpoints(3, args.seed + 1, device)
    freq_indices = torch.sort(torch.randperm(freqs.numel(), device=device)[:12])[0]

    ok = True
    for sign in (1.0, -1.0):
        for i in range(3):
            theta = thetas[i:i + 1].unsqueeze(0)
            phi = phis[i:i + 1].unsqueeze(0)
            rx_pos, tx_pos = get_array_pos(theta, phi, 10.0, spacing, 4, 4, device)
            ref = direct_forward(freqs, kvector, rx_pos, tx_pos, pos, weights, phase_sign=sign)
            pred = range_forward_operator(
                freqs, kvector, rx_pos, tx_pos, pos, weights,
                phase_sign=sign, pair_chunk=5, point_chunk=17, compute_dtype=torch.float64,
            )
            pred_subset = range_forward_operator(
                freqs, kvector, rx_pos, tx_pos, pos, weights,
                phase_sign=sign, freq_indices=freq_indices, pair_chunk=5, point_chunk=17,
                compute_dtype=torch.float64,
            )
            err_full = rel_l2(pred, ref)
            err_subset = rel_l2(pred_subset, ref[freq_indices])
            passed = err_full < 1e-9 and err_subset < 1e-9
            ok &= passed
            print(f"  sign={sign:+.0f} viewpoint={i}: full={err_full:.3e} subset={err_subset:.3e} "
                  f"{'PASS' if passed else 'FAIL'}")
    return ok


def stage_b(args):
    print("\nStage B: realistic near-field scale (GPU)")
    if not torch.cuda.is_available():
        print("  SKIP: CUDA is not available in this environment")
        return True
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    freqs = torch.linspace(95e9, 105e9, 1000, dtype=torch.float64, device=device)
    kvector = get_kvector(freqs, cc)
    freq_indices = torch.sort(torch.randperm(freqs.numel(), device=device)[:64])[0]
    theta_vals, phi_vals = real_sphere_viewpoints(4, args.real_viewpoints_dir, args.seed, device)

    def run_case(pos, label):
        weights = torch.randn(pos.shape[0], dtype=torch.float64, device=device) + 1j * torch.randn(
            pos.shape[0], dtype=torch.float64, device=device
        )
        all_pred, all_ref = [], []
        for theta_val, phi_val in zip(theta_vals, phi_vals):
            theta = theta_val.view(1, 1)
            phi = phi_val.view(1, 1)
            rx_pos, tx_pos = get_array_pos(theta, phi, 10.0, spacing, 16, 16, device)
            ref = direct_forward(freqs[freq_indices], kvector[freq_indices], rx_pos, tx_pos, pos, weights)
            pred = range_forward_operator(
                freqs, kvector, rx_pos, tx_pos, pos, weights, freq_indices=freq_indices,
                pair_chunk=args.pair_chunk, point_chunk=args.point_chunk, compute_dtype=torch.float64,
            )
            all_pred.append(pred.reshape(-1))
            all_ref.append(ref.reshape(-1))
        pred_cat = torch.cat(all_pred)
        ref_cat = torch.cat(all_ref)
        err64 = global_rel_mse(pred_cat, ref_cat)
        pred32 = range_forward_operator(
            freqs.float(), get_kvector(freqs.float(), cc), rx_pos, tx_pos, pos.float(), weights.to(torch.complex64),
            freq_indices=freq_indices, pair_chunk=args.pair_chunk, point_chunk=args.point_chunk,
            compute_dtype=torch.float32,
        )
        err32 = global_rel_mse(pred32.to(torch.complex128), ref)
        print(f"  {label}: fp64 global_rel_mse={err64:.3e}, fp32 last-view={err32:.3e}")
        return err64 < 1e-8 and err32 < 1e-3

    grid = generate_dynamic_grid(24, extent, device, jitter=False).reshape(-1, 3).to(torch.float64)
    ok_grid = run_case(grid, "grid-13824")
    pos_100k = (torch.rand(100_000, 3, dtype=torch.float64, device=device) - 0.5) * 6.0
    ok_100k = run_case(pos_100k, "random-100000")
    return ok_grid and ok_100k


def stage_c(args):
    print("\nStage C: gradients (CPU)")
    device = torch.device("cpu")
    torch.manual_seed(args.seed + 2)

    freqs = torch.linspace(95e9, 105e9, 16, dtype=torch.float64, device=device)
    kvector = get_kvector(freqs, cc)
    theta = torch.tensor([[1.0]], dtype=torch.float64, device=device)
    phi = torch.tensor([[0.3]], dtype=torch.float64, device=device)
    rx_pos, tx_pos = get_array_pos(theta, phi, 10.0, spacing, 2, 2, device)

    pos0 = (torch.rand(6, 3, dtype=torch.float64, device=device) - 0.5).requires_grad_()
    w_re0 = torch.randn(6, dtype=torch.float64, device=device).requires_grad_()
    w_im0 = torch.randn(6, dtype=torch.float64, device=device).requires_grad_()

    def loss_for_gradcheck(pos, w_re, w_im):
        y = range_forward_operator(
            freqs, kvector, rx_pos, tx_pos, pos, torch.complex(w_re, w_im),
            pair_chunk=3, point_chunk=4, compute_dtype=torch.float64,
        )
        return (y.real.square() + y.imag.square()).sum()

    gradcheck_ok = torch.autograd.gradcheck(
        loss_for_gradcheck, (pos0, w_re0, w_im0), eps=1e-6, atol=1e-5, rtol=1e-4
    )
    print(f"  gradcheck: {'PASS' if gradcheck_ok else 'FAIL'}")

    pos = (torch.rand(6, 3, dtype=torch.float64, device=device) - 0.5).requires_grad_()
    w_re = torch.randn(6, dtype=torch.float64, device=device).requires_grad_()
    w_im = torch.randn(6, dtype=torch.float64, device=device).requires_grad_()
    target = torch.randn(16, 2, 2, dtype=torch.float64, device=device) + 1j * torch.randn(
        16, 2, 2, dtype=torch.float64, device=device
    )

    y_range = range_forward_operator(
        freqs, kvector, rx_pos, tx_pos, pos, torch.complex(w_re, w_im),
        pair_chunk=3, point_chunk=4, compute_dtype=torch.float64,
    )
    loss_range = (y_range - target).abs().square().sum()
    grads_range = torch.autograd.grad(loss_range, (pos, w_re, w_im), retain_graph=False)

    y_brute = forward_operator_lessparallel(
        freqs, kvector, rx_pos, tx_pos, pos, torch.complex(w_re, w_im),
        artificial_gain=1.0, p_spectrum=None, range_model="sum2", omega_scaling="unity",
        center_freq_hz=None, phase_sign=1.0,
    )
    loss_brute = (y_brute.to(torch.complex128) - target).abs().square().sum()
    grads_brute = torch.autograd.grad(loss_brute, (pos, w_re, w_im), retain_graph=False)

    names = ("pos", "w_re", "w_im")
    rels = []
    for name, g_range, g_brute in zip(names, grads_range, grads_brute):
        err = rel_l2(g_range, g_brute)
        rels.append(err)
        print(f"  gradient match {name}: {err:.3e}")
    match_ok = max(rels) < 1e-5
    print(f"  gradient match: {'PASS' if match_ok else 'FAIL'}")
    return bool(gradcheck_ok) and match_ok


def stage_d(args):
    print("\nStage D: adjoint")
    device = torch.device("cpu")
    torch.manual_seed(args.seed + 3)
    freqs = torch.linspace(95e9, 105e9, 32, dtype=torch.float64, device=device)
    kvector = get_kvector(freqs, cc)
    theta = torch.tensor([[1.2]], dtype=torch.float64, device=device)
    phi = torch.tensor([[0.4]], dtype=torch.float64, device=device)
    rx_pos, tx_pos = get_array_pos(theta, phi, 10.0, spacing, 4, 4, device)
    pos = (torch.rand(64, 3, dtype=torch.float64, device=device) - 0.5) * 6.0
    weights = torch.randn(64, dtype=torch.float64, device=device) + 1j * torch.randn(64, dtype=torch.float64, device=device)
    y = torch.randn(32, 4, 4, dtype=torch.float64, device=device) + 1j * torch.randn(32, 4, 4, dtype=torch.float64, device=device)

    ax = range_forward_operator(freqs, kvector, rx_pos, tx_pos, pos, weights, pair_chunk=5, point_chunk=17)
    ahy = range_adjoint_operator(freqs, kvector, rx_pos, tx_pos, pos, y, pair_chunk=5, point_chunk=17)
    lhs = (ax.conj() * y).sum()
    rhs = (weights.conj() * ahy).sum()
    dot_err = float((lhs - rhs).abs() / (ax.norm() * y.norm()).clamp_min(EPS))
    print(f"  dot test: {dot_err:.3e}")

    ref_ahy = direct_adjoint(freqs, kvector, rx_pos, tx_pos, pos, y, phase_sign=1.0)
    brute_err = rel_l2(ahy, ref_ahy)
    print(f"  vs direct adjoint: {brute_err:.3e}")

    w_re = weights.real.detach().clone().requires_grad_()
    w_im = weights.imag.detach().clone().requires_grad_()
    ax2 = range_forward_operator(freqs, kvector, rx_pos, tx_pos, pos, torch.complex(w_re, w_im), pair_chunk=5, point_chunk=17)
    scalar = (ax2.conj() * y).sum().real
    grad_re, grad_im = torch.autograd.grad(scalar, (w_re, w_im))
    vjp = torch.complex(grad_re, grad_im)
    vjp_err = rel_l2(vjp, ahy)
    print(f"  autograd VJP: {vjp_err:.3e}")

    return dot_err < 1e-10 and brute_err < 1e-5 and vjp_err < 1e-10


def stage_e(args):
    print("\nStage E: benchmark (GPU, report-only)")
    if not torch.cuda.is_available():
        print("  SKIP: CUDA is not available in this environment")
        return True
    device = torch.device(args.device)
    torch.manual_seed(args.seed + 4)
    print("| N | nf | dtype | forward_s | forward_backward_s | max_mem_MB |")
    print("|---:|---:|---|---:|---:|---:|")
    for n_points in (13_824, 100_000, 1_000_000):
        for nf_sel in (200, 1000):
            freqs = torch.linspace(95e9, 105e9, 1000, dtype=torch.float64, device=device)
            kvector = get_kvector(freqs, cc)
            freq_indices = torch.arange(0, 1000, max(1, 1000 // nf_sel), device=device)[:nf_sel]
            theta = torch.tensor([[1.0]], dtype=torch.float64, device=device)
            phi = torch.tensor([[0.4]], dtype=torch.float64, device=device)
            rx_pos, tx_pos = get_array_pos(theta, phi, 10.0, spacing, 16, 16, device)
            pos = (torch.rand(n_points, 3, dtype=torch.float64, device=device) - 0.5) * 6.0
            for dtype in (torch.float64, torch.float32):
                pos_case = pos.to(dtype).detach().requires_grad_(True)
                weights = (torch.randn(n_points, dtype=dtype, device=device) + 1j * torch.randn(
                    n_points, dtype=dtype, device=device
                )).requires_grad_(True)
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize()
                t0 = time.time()
                pred = range_forward_operator(
                    freqs.to(dtype), get_kvector(freqs.to(dtype), cc), rx_pos.to(dtype), tx_pos.to(dtype),
                    pos_case, weights, freq_indices=freq_indices, pair_chunk=args.pair_chunk,
                    point_chunk=args.point_chunk, compute_dtype=dtype,
                )
                torch.cuda.synchronize()
                t1 = time.time()
                loss = pred.abs().square().mean()
                loss.backward()
                torch.cuda.synchronize()
                t2 = time.time()
                mem_mb = torch.cuda.max_memory_allocated(device) / 1024**2
                print(f"| {n_points} | {nf_sel} | {str(dtype).replace('torch.', '')} | "
                      f"{t1 - t0:.3f} | {t2 - t0:.3f} | {mem_mb:.1f} |")
    return True


def stage_f(args):
    print("\nStage F: float32-frequency pipeline (CPU)")
    device = torch.device("cpu")
    files = sorted(glob.glob(os.path.join(args.real_viewpoints_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {args.real_viewpoints_dir}")
    dataset = CSVSimulationDataset([files[0]], device=device)
    freqs_tensor, dphi_tensor, dtheta_tensor, _, _ = dataset[0]
    kvector = get_kvector(freqs_tensor, cc)
    nf = freqs_tensor.shape[0]
    ideal = torch.linspace(float(freqs_tensor[0]), float(freqs_tensor[-1]), nf, dtype=torch.float64, device=device)
    df = ideal[1] - ideal[0]
    ratio = float((freqs_tensor.double() - ideal).abs().max() / df.abs())

    torch.manual_seed(args.seed + 5)
    pos = (torch.rand(64, 3, dtype=torch.float64, device=device) - 0.5) * 6.0
    weights = torch.randn(64, dtype=torch.float64, device=device) + 1j * torch.randn(64, dtype=torch.float64, device=device)
    rx_pos, tx_pos = get_array_pos(dtheta_tensor.unsqueeze(0), dphi_tensor.unsqueeze(0), 10.0, spacing, 4, 4, device)

    pred = range_forward_operator(
        freqs_tensor, kvector, rx_pos, tx_pos, pos, weights,
        pair_chunk=5, point_chunk=17, compute_dtype=torch.float64,
    )
    print(f"  accept real float32 frequencies: max|freqs-ideal|/df={ratio:.3e}")

    displaced = freqs_tensor.clone()
    displaced[nf // 2] += (0.05 * df).to(displaced.dtype)
    permuted = freqs_tensor[torch.randperm(nf)]
    rejected = 0
    for label, bad_freqs in (("displaced", displaced), ("permuted", permuted)):
        try:
            range_forward_operator(
                bad_freqs, get_kvector(bad_freqs, cc), rx_pos, tx_pos, pos, weights,
                pair_chunk=5, point_chunk=17, compute_dtype=torch.float64,
            )
            print(f"  reject {label}: FAIL")
        except ValueError:
            rejected += 1
            print(f"  reject {label}: PASS")

    ref_ideal = direct_forward(ideal, get_kvector(ideal, cc), rx_pos, tx_pos, pos, weights)
    ideal_err = rel_l2(pred, ref_ideal)
    ref_raw = direct_forward(freqs_tensor.double(), get_kvector(freqs_tensor.double(), cc), rx_pos, tx_pos, pos, weights)
    raw_vs_ideal = rel_l2(ref_raw, ref_ideal)
    print(f"  operator vs reconstructed ideal: {ideal_err:.3e}")
    print(f"  raw float32 direct vs ideal direct: {raw_vs_ideal:.3e}")
    return rejected == 2 and ideal_err < 1e-9 and ideal_err * 1e3 < raw_vs_ideal


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--stages", default="A,C,D,F",
                   help="Comma-separated stages from A,B,C,D,E,F or 'all'. GPU stages B/E skip if CUDA is absent.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--real-viewpoints-dir", default=REAL_SPHERE_CSV_DIR)
    p.add_argument("--pair-chunk", type=int, default=64)
    p.add_argument("--point-chunk", type=int, default=262144)
    args = p.parse_args()

    selected = ["A", "B", "C", "D", "E", "F"] if args.stages.lower() == "all" else [
        s.strip().upper() for s in args.stages.split(",") if s.strip()
    ]
    runners = {"A": stage_a, "B": stage_b, "C": stage_c, "D": stage_d, "E": stage_e, "F": stage_f}
    ok = True
    for stage in selected:
        if stage not in runners:
            raise ValueError(f"Unknown stage {stage}")
        stage_ok = runners[stage](args)
        print(f"Stage {stage}: {'PASS' if stage_ok else 'FAIL'}")
        ok &= stage_ok
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
