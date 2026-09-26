"""Gate for grow()'s threshold modes and selectivity report (2026-08-02).

Run: PYTHONPATH=. python scripts/validate_grow_threshold.py

Checks that --grow-threshold-mode quantile really selects the requested
fraction (against torch.quantile, on a lognormal / a tight <1-decade
distribution / a bimodal one), that the printed selectivity report is exact,
and that grow()/grow_angular() return the 3-tuple train.py expects. The
"tight" case is the one that matters: it reproduces the Round 3 pathology
where --grow-threshold 0.1 in relmax mode grew 110592/110592 voxels.

The sharded twin is covered separately by scripts/validate_distributed_scene.py
(block E), which must be run at world_size >= 2.
"""
import torch
from rift.sparse_scene import (SHVoxelGridScene, _threshold_for_fraction,
                               _global_frac_ge)

dev = "cpu"
torch.manual_seed(0)

# --- 1. quantile resolution against torch.quantile -------------------------
print("--- bisected quantile vs torch.quantile")
for name, vals in [("lognormal", torch.exp(torch.randn(200000) * 2.0)),
                   ("tight (R3-like, <1 decade)", 0.5 + torch.rand(200000) * 0.5),
                   ("bimodal", torch.cat([torch.rand(190000) * 0.01,
                                          torch.rand(10000) * 1.0]))]:
    gmax = vals.max()
    for q in (0.5, 0.1, 0.01, 0.001):
        t = _threshold_for_fraction(vals, gmax, q, vals.numel(), vals.device, False)
        got = float((vals >= t).float().mean())
        ref = float(torch.quantile(vals, 1 - q))
        print(f"  {name:28s} q={q:<6} thresh/max={float(t/gmax):.5f} "
              f"(exact {ref/float(gmax):.5f})  actually grown={got:.5f}")
        assert abs(got - q) <= max(0.02 * q, 2.0 / vals.numel()), \
            f"selected {got}, requested {q}"

# --- 2. report values are exact -------------------------------------------
print("--- report exactness")
vals = torch.exp(torch.randn(50000))
ts = torch.tensor([0.9, 0.5, 0.1, 0.03], dtype=torch.float64) * vals.max().double()
fr = _global_frac_ge(vals, ts, vals.numel(), vals.device, False)
for t, f in zip((0.9, 0.5, 0.1, 0.03), fr):
    # count exactly: an fp32 .mean() over 50k elements is itself only ~1e-8 good
    truth = int((vals.double() >= t * vals.max().double()).sum()) / vals.numel()
    print(f"  relmax {t:<5} -> frac {float(f):.5f} (truth {truth:.5f})")
    assert abs(float(f) - truth) < 1e-9

# --- 3. end-to-end grow() on a real scene ----------------------------------
print("--- SHVoxelGridScene.grow")
for mode, thr, want in [("relmax", 0.1, None), ("quantile", 0.05, 0.05),
                        ("quantile", 0.2, 0.2)]:
    s = SHVoxelGridScene(16, 1.5, dev, max_degree=4, init_degree=0)
    n = s.w_re.shape[:3].numel()
    # R3-like tight gradient field: uniform in [0.5, 1.0], under one decade
    s.grad_accum += 0.5 + torch.rand(16, 16, 16, device=dev) * 0.5
    s.grad_accum_count += 1
    g, a, rep = s.grow(threshold_fraction=thr, mode=mode)
    print(f"  mode={mode:9s} thr={thr:<5} grew {g}/{a} ({g/n:.3f})")
    print(f"    {rep}")
    if want is not None:
        assert abs(g / n - want) < 0.25 * want, f"wanted ~{want}, got {g/n}"
    else:
        assert g == n, "relmax 0.1 on a tight field should still grow everything"

# --- 4. grow_angular still returns a 3-tuple + report ----------------------
s = SHVoxelGridScene(8, 1.5, dev, max_degree=4, init_degree=1)
s.w_re.data[..., 0] = 1.0
s.w_re.data[..., 1:4] = 0.3
g, a, rep = s.grow_angular(tail_ratio_threshold=0.05)
print(f"--- grow_angular grew {g}/{a}\n    {rep}")
assert g == a and rep

print("\nALL GROW GATES PASS")
