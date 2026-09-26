#!/usr/bin/env python
"""CPU contract gate for the two Round-8 mechanisms: the polar-cap
(angular-extrapolation) validation split, and --adam-eps.

Run at the repo root:
    module load anaconda3 && conda activate RIFT
    python scripts/validate_extrapolation_split.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rift.npz_dataset import polar_cap_split  # noqa: E402

PASS, FAIL = 0, 0


def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


def fibonacci_sphere(n):
    """Same construction as the datasets: elevation-ordered by build."""
    i = np.arange(n, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1) * 10.0


def main():
    print("=" * 74)
    print("1. polar cap split -- geometry")
    print("=" * 74)
    pos = fibonacci_sphere(2000)
    unit = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    axis = np.array([1.0, 1.0, 1.0])
    rng = np.random.default_rng(42)
    tr, va, te, half = polar_cap_split(pos, 200, axis, 1800, 0, rng)

    check("val has exactly num_val entries", len(va) == 200, f"got {len(va)}")
    check("train has exactly num_train entries", len(tr) == 1800, f"got {len(tr)}")
    check("train and val are disjoint", len(set(tr) & set(va)) == 0)
    check("test empty when num_test=0", len(te) == 0)
    check("indices are a valid subset of the view set",
          set(tr) | set(va) <= set(range(2000)))

    a = axis / np.linalg.norm(axis)
    proj_val, proj_train = unit[va] @ a, unit[tr] @ a
    check("every held-out view is closer to the axis than every training view",
          proj_val.min() > proj_train.max(),
          f"val min {proj_val.min():.6f} vs train max {proj_train.max():.6f}")
    check("reported half-angle matches the outermost held-out view",
          abs(half - np.degrees(np.arccos(proj_val.min()))) < 1e-9)
    # The whole point of the split: held-out directions must sit OUTSIDE the
    # convex hull of the training directions.  For a cap that is equivalent to
    # the statement above, and it is what a random split fails.
    rand_val = rng.permutation(2000)[:200]
    rand_train = np.setdiff1d(np.arange(2000), rand_val)
    check("a RANDOM split does NOT have that property (control)",
          not ((unit[rand_val] @ a).min() > (unit[rand_train] @ a).max()),
          "a random held-out set should interleave with training views, not sit outside them")

    print()
    print("=" * 74)
    print("2. polar cap split -- ordering and determinism")
    print("=" * 74)
    # Fibonacci indices are elevation-ordered, so a cap-distance-ordered training
    # stream would sweep the sphere instead of being i.i.d.
    order_corr = np.corrcoef(np.arange(len(tr)), proj_train)[0, 1]
    check("training views are NOT ordered by distance from the cap",
          abs(order_corr) < 0.1, f"corr(position in stream, alignment) = {order_corr:.4f}")
    tr2, va2, _, _ = polar_cap_split(pos, 200, axis, 1800, 0, np.random.default_rng(42))
    check("same seed reproduces the split exactly",
          np.array_equal(tr, tr2) and np.array_equal(va, va2))
    tr3, va3, _, _ = polar_cap_split(pos, 200, axis, 1800, 0, np.random.default_rng(7))
    check("val set is seed-INDEPENDENT (geometry, not RNG)", np.array_equal(va, va3))
    check("train order is seed-DEPENDENT", not np.array_equal(tr, tr3))

    for ax, nm in [([0, 0, 1], "+z"), ([0, 0, -1], "-z"), ([1, 0, 0], "+x")]:
        t, v, _, h = polar_cap_split(pos, 200, np.array(ax, float), 1800, 0,
                                     np.random.default_rng(42))
        u = unit @ (np.array(ax, float) / np.linalg.norm(ax))
        check(f"axis {nm}: cap is contiguous and disjoint from train",
              u[v].min() > u[t].max() and len(set(t) & set(v)) == 0)
    check("cap axis need not be normalized", np.array_equal(
        polar_cap_split(pos, 200, np.array([5.0, 5.0, 5.0]), 1800, 0,
                        np.random.default_rng(42))[1], va))

    print()
    print("=" * 74)
    print("3. --adam-eps: the throttle it removes")
    print("=" * 74)
    # Adam's realised step is lr*m/(sqrt(v)+eps).  Reproduce the throttle on a
    # scalar whose gradient matches the MEASURED scene gradient scale.
    for g_scale, label in [(5.9e-12, "converged scene"), (1.1e-10, "near-empty scene")]:
        steps = {}
        for eps in (1e-8, 1e-20):
            p = torch.zeros(1, requires_grad=True)
            opt = torch.optim.AdamW([{"params": [p], "lr": 3e-3, "eps": eps}])
            for _ in range(40):                     # let the moments settle
                p.grad = torch.full_like(p, g_scale)
                opt.step()
            steps[eps] = abs(p.item()) / 40
        ratio = steps[1e-20] / max(steps[1e-8], 1e-300)
        check(f"{label} (|g|={g_scale:g}): eps=1e-20 steps >> eps=1e-8",
              ratio > 10, f"ratio {ratio:.1f}x")
        print(f"         mean |step| {steps[1e-8]:.3e} (eps 1e-8) vs "
              f"{steps[1e-20]:.3e} (eps 1e-20) -> {ratio:.0f}x")

    p = torch.zeros(1, requires_grad=True)
    opt = torch.optim.AdamW([{"params": [p], "lr": 3e-3, "eps": 1e-8}])
    for _ in range(40):
        p.grad = torch.full_like(p, 1.0)            # a healthy gradient
        opt.step()
    healthy = abs(p.item()) / 40
    check("a HEALTHY gradient (|g|=1) is not throttled by the default eps",
          healthy > 0.9 * 3e-3, f"mean step {healthy:.3e} vs lr 3e-3")

    print()
    print("=" * 74)
    print(f"{PASS}/{PASS + FAIL} checks passed")
    print("=" * 74)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
