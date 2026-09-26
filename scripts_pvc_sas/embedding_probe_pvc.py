#!/usr/bin/env python
"""Diagnostic for the hash-grid encoder backward on XPU (Package G, G0 follow-up).

The device tests found the ``SHSASField`` hash-table gradients 5-35 % off the
CPU values and not repeatable run to run, while every other sonar gradient
matched. The tables are ``nn.Embedding`` modules gathered by ``table(indices)``
(``F.embedding``); this probe isolates that backward on the device and compares
three mathematically identical gathers (``F.embedding``, advanced indexing
``weight[idx]``, ``index_select``) against an fp64 CPU reference, with heavy
duplicate indices as the hash grid produces, and reports bitwise repeatability.
It then runs the real ``HashGridEncoder`` (production configuration) with each
gather variant. Exit 0 always; the JSON report is the deliverable.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rift.radar_fields import HashGridEncoder  # noqa: E402
from rift_pvc import accelerator  # noqa: E402

GATHERS = {
    "F.embedding": lambda w, idx: F.embedding(idx, w),
    "weight[idx]": lambda w, idx: w[idx],
    "index_select": lambda w, idx: torch.index_select(w, 0, idx.reshape(-1)).reshape(*idx.shape, w.shape[1]),
}


def rel(a, b):
    a, b = a.detach().cpu().double(), b.detach().cpu().double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def grad_of(gather, weight, idx, upstream):
    w = weight.detach().clone().requires_grad_(True)
    out = gather(w, idx)
    (out * upstream).sum().backward()
    return w.grad.detach()


def repeatable(fn, n=3):
    ref = fn().cpu()
    return all(torch.equal(fn().cpu(), ref) for _ in range(n - 1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    device = accelerator.device()
    results = []
    gen = torch.Generator().manual_seed(0)
    for entries, n_idx, dup in ((4096, 8192, True), (2 ** 19, 110592 * 8, True), (2 ** 19, 110592 * 8, False), (65536, 4096, False)):
        weight = torch.rand(entries, 2, generator=gen) * 2e-4 - 1e-4
        if dup:
            idx = torch.randint(0, max(entries // 64, 8), (n_idx // 8, 8), generator=gen)  # heavy duplicates
        else:
            idx = torch.randperm(entries, generator=gen)[:n_idx].reshape(-1, 8) if n_idx <= entries else \
                torch.randint(0, entries, (n_idx // 8, 8), generator=gen)
        upstream = torch.randn(idx.shape[0], 8, 2, generator=gen)
        ref64 = grad_of(GATHERS["F.embedding"], weight.double(), idx, upstream.double())
        for name, gather in GATHERS.items():
            cpu32 = grad_of(gather, weight, idx, upstream)
            dev = lambda: grad_of(gather, weight.to(device), idx.to(device), upstream.to(device))
            accelerator.synchronize(); t = time.perf_counter(); g = dev(); accelerator.synchronize(); ms = (time.perf_counter() - t) * 1e3
            fwd_dev = gather(weight.to(device), idx.to(device))
            results.append({"case": f"entries={entries} n_idx={n_idx} dup={dup}", "gather": name,
                            "fwd_rel_vs_cpu": rel(fwd_dev, gather(weight, idx)),
                            "grad_rel_cpu32_vs_fp64": rel(cpu32, ref64), "grad_rel_xpu32_vs_fp64": rel(g, ref64),
                            "grad_rel_xpu_vs_cpu": rel(g, cpu32), "xpu_repeatable": repeatable(dev), "ms": ms,
                            "nonzero_rows_cpu": int((cpu32.abs().sum(1) > 0).sum()), "nonzero_rows_xpu": int((g.abs().sum(1) > 0).sum())})
            print("PROBE", json.dumps(results[-1]), flush=True)
    # the real encoder, production configuration, three gather variants
    torch.manual_seed(0)
    enc = HashGridEncoder(n_levels=16, n_features_per_level=2, base_resolution=16, final_resolution=4096, log2_hashmap_size=19)
    pts = torch.rand(110592, 3, generator=gen)

    def encoder_forward(encoder, unit_xyz, gather):
        xyz = unit_xyz.clamp(0.0, 1.0 - 1.0e-7)
        corners = encoder.corners.to(xyz.device)
        encoded = []
        for level, (resolution, table) in enumerate(zip(encoder.resolutions, encoder.tables)):
            scaled = xyz * float(resolution)
            base = torch.floor(scaled).long()
            frac = scaled - base.to(scaled.dtype)
            corner_coords = base[:, None, :] + corners[None, :, :]
            indices = encoder._hash(corner_coords, table.num_embeddings)
            features = gather(table.weight, indices)
            corner_f = corners.to(frac.dtype)[None, :, :]
            weights = torch.where(corner_f.bool(), frac[:, None, :], 1.0 - frac[:, None, :]).prod(dim=-1)
            encoded.append((features * weights[..., None]).sum(dim=1))
        return torch.cat(encoded, dim=-1)

    upstream = torch.randn(110592, 32, generator=gen)

    def table_grads(encoder, xyz, gather, up):
        encoder.zero_grad(set_to_none=True)
        (encoder_forward(encoder, xyz, gather) * up).sum().backward()
        return torch.cat([t.weight.grad.reshape(-1) for t in encoder.tables])

    enc64 = HashGridEncoder(n_levels=16, n_features_per_level=2, base_resolution=16, final_resolution=4096, log2_hashmap_size=19)
    enc64.load_state_dict(enc.state_dict()); enc64.double()
    ref64 = table_grads(enc64, pts, GATHERS["F.embedding"], upstream.double())
    original_cpu = table_grads(enc, pts, GATHERS["F.embedding"], upstream)
    enc_dev = HashGridEncoder(n_levels=16, n_features_per_level=2, base_resolution=16, final_resolution=4096, log2_hashmap_size=19)
    enc_dev.load_state_dict(enc.state_dict()); enc_dev.to(device)
    for name, gather in GATHERS.items():
        cpu32 = table_grads(enc, pts, gather, upstream)
        dev = lambda: table_grads(enc_dev, pts.to(device), gather, upstream.to(device))
        accelerator.synchronize(); t = time.perf_counter(); g = dev(); accelerator.synchronize(); ms = (time.perf_counter() - t) * 1e3
        fwd_rel = rel(encoder_forward(enc_dev, pts.to(device), gather), encoder_forward(enc, pts, gather))
        results.append({"case": "HashGridEncoder production config, 110592 points", "gather": name, "fwd_rel_vs_cpu": fwd_rel,
                        "grad_rel_cpu32_vs_fp64": rel(cpu32, ref64), "grad_rel_xpu32_vs_fp64": rel(g, ref64),
                        "grad_rel_xpu_vs_cpu": rel(g, cpu32), "grad_rel_cpu_variant_vs_original_cpu": rel(cpu32, original_cpu),
                        "xpu_repeatable": repeatable(dev), "ms": ms})
        print("PROBE", json.dumps(results[-1]), flush=True)
    # does torch.use_deterministic_algorithms change the F.embedding backward on the device?
    try:
        torch.use_deterministic_algorithms(True)
        dev = lambda: table_grads(enc_dev, pts.to(device), GATHERS["F.embedding"], upstream.to(device))
        g = dev()
        results.append({"case": "HashGridEncoder, use_deterministic_algorithms(True)", "gather": "F.embedding",
                        "grad_rel_xpu32_vs_fp64": rel(g, ref64), "xpu_repeatable": repeatable(dev)})
    except Exception as exc:  # noqa: BLE001
        results.append({"case": "HashGridEncoder, use_deterministic_algorithms(True)", "error": f"{type(exc).__name__}: {exc}"})
    finally:
        torch.use_deterministic_algorithms(False)
    print("PROBE", json.dumps(results[-1]), flush=True)
    Path(args.output).write_text(json.dumps({"accelerator": accelerator.describe(), "results": results}, indent=2) + "\n")
    print("EMBEDDING_PROBE_DONE", args.output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
