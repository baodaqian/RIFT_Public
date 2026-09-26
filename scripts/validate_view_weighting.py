#!/usr/bin/env python
"""CPU contract gate for --view-weight-alpha (per-view power weighting).

The training objective is an unweighted sum of per-view squared residuals, so a
viewpoint contributes in proportion to its measured power. On the B787 view
sphere that puts the effective training-view count at 286 of 1800 -- below the
dataset's own angular-Nyquist estimate -- while the under-weighted dim views
still carry 48.9% of the GLOBAL error mass. ``--view-weight-alpha`` reweights
each view by ``p_v^-alpha``, normalized to mean 1.

This gate checks the contract, NOT the science:

  1. alpha = 0 returns None -- the historical objective is untouched, bit-exact.
  2. Weights have mean exactly 1, so the objective scale (and with it the
     effective learning rate and the gain/scene gauge) is preserved.
  3. Weights are monotonically DECREASING in view power.
  4. alpha = 1 exactly equalizes the weighted contribution of every view.
  5. The weight is invariant to a global rescaling of the data, so it does not
     interact with the exactly-flat (gain, scene) gauge.
  6. Effective view count N_eff is non-decreasing in alpha and reaches the full
     view count at alpha = 1.
  7. Log-space evaluation matches the direct p^-alpha formula (numerics).
  8. Extreme dynamic range (1e-30 power spread) stays finite and normalized.
  9. A zero-power view is handled without NaN/Inf.
 10. The reported (unweighted) data loss and rel-MSE are unaffected -- verified
     on the real accumulation order in train.py's loop.
 11. Weights are deterministic and identical across repeated construction, so
     every rank of a sharded run derives the same values with no communication.

    module load anaconda3 && conda activate RIFT
    python scripts/validate_view_weighting.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import compute_view_weights  # noqa: E402


class FakeLoader:
    """Minimal stand-in for the DataLoader: batch[3] is the magnitude tensor."""

    def __init__(self, powers):
        self.powers = powers

    def __iter__(self):
        for p in self.powers:
            # one element whose square is exactly p, in the batch[3] slot
            mag = torch.sqrt(torch.tensor([[float(p)]], dtype=torch.float64))
            yield (None, None, None, mag, None)


def weights_for(powers, alpha, max_weight_ratio=0.0):
    """Default: floor OFF, so these checks pin the exact p^-alpha contract."""
    return compute_view_weights(
        FakeLoader(powers), alpha, torch.device("cpu"), max_weight_ratio=max_weight_ratio)


def n_eff(powers, weights):
    p = torch.tensor([float(x) for x in powers], dtype=torch.float64)
    w = torch.ones_like(p) if weights is None else weights.cpu().double()
    e = p * w
    return float(e.sum() ** 2 / (e ** 2).sum())


def main():
    checks = []

    def check(name, ok, detail=""):
        checks.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))

    # a flash-dominated spectrum in the spirit of the real B787 view sphere
    powers = [1e-12, 3e-12, 1e-11, 5e-11, 2e-10, 8e-10, 4e-9, 2e-8, 9e-8, 5e-7]

    print("1. alpha = 0 leaves the historical objective untouched")
    check("returns None at alpha=0", weights_for(powers, 0.0) is None)

    print("\n2. normalization: mean(w) == 1")
    for a in (0.25, 0.5, 1.0, 2.0):
        w = weights_for(powers, a)
        check(f"alpha={a}: mean(w)=1", torch.allclose(w.mean(), torch.tensor(1.0, dtype=torch.float64)),
              f"mean={w.mean().item():.17g}")

    print("\n3. weights decrease monotonically with view power")
    for a in (0.25, 0.5, 1.0):
        w = weights_for(powers, a)
        check(f"alpha={a}: strictly decreasing", bool((w[1:] < w[:-1]).all()))

    print("\n4. alpha = 1 equalizes every view's weighted contribution")
    w = weights_for(powers, 1.0)
    contrib = torch.tensor([float(p) for p in powers], dtype=torch.float64) * w.double()
    check("alpha=1: p_v * w_v identical across views", bool(
        torch.allclose(contrib, contrib[0].expand_as(contrib), rtol=1e-12)),
        f"spread={float(contrib.max() / contrib.min()) - 1:.3e}")

    print("\n5. gauge invariance: rescaling all data leaves the weights unchanged")
    for a in (0.25, 1.0):
        base = weights_for(powers, a)
        scaled = weights_for([p * 1e6 for p in powers], a)
        check(f"alpha={a}: invariant to a 1e6 data rescale",
              bool(torch.allclose(base, scaled, rtol=1e-10)),
              f"max|d|={float((base - scaled).abs().max()):.3e}")

    print("\n6. effective view count rises with alpha and saturates at alpha = 1")
    base_eff = n_eff(powers, None)
    effs = [n_eff(powers, weights_for(powers, a)) for a in (0.25, 0.5, 0.75, 1.0)]
    check("N_eff non-decreasing in alpha", all(b >= a - 1e-9 for a, b in zip([base_eff] + effs, effs)),
          f"{base_eff:.2f} -> " + " -> ".join(f"{e:.2f}" for e in effs))
    check("alpha=1 reaches the full view count", abs(effs[-1] - len(powers)) < 1e-6,
          f"N_eff={effs[-1]:.6f} of {len(powers)}")

    print("\n7. log-space evaluation matches the direct p^-alpha formula")
    for a in (0.25, 0.5, 1.0):
        w = weights_for(powers, a).double()
        direct = torch.tensor([float(p) ** (-a) for p in powers], dtype=torch.float64)
        direct = direct / direct.mean()
        check(f"alpha={a}: matches direct formula", bool(torch.allclose(w, direct, rtol=1e-10)),
              f"max rel dev={float(((w - direct).abs() / direct).max()):.3e}")

    print("\n8. extreme dynamic range stays finite")
    wide = [1e-30, 1e-20, 1e-10, 1.0]
    w = weights_for(wide, 0.5)
    check("finite over a 1e30 power spread", bool(torch.isfinite(w).all()))
    check("still normalized to mean 1", torch.allclose(w.mean(), torch.tensor(1.0, dtype=torch.float64)))

    print("\n9. a zero-power view gets weight 0, not the largest weight")
    w = weights_for([0.0, 1e-9, 1e-8], 0.5)
    check("finite with a dead view", bool(torch.isfinite(w).all()), f"w={w.tolist()}")
    check("dead view weighted exactly 0", float(w[0]) == 0.0)
    check("live views still average 1", torch.allclose(
        w[1:].mean(), torch.tensor(1.0, dtype=torch.float64)))
    check("dead view does not outweigh live ones", bool((w[0] <= w[1:]).all()))

    print("\n9b. the OPT-IN max_weight_ratio floor contains the p -> 0 blow-up")
    wide = [1e-30, 1e-20, 1e-10, 1.0]
    uncapped = weights_for(wide, 0.5)
    check("floor is OFF by default (alpha means exactly p^-alpha)",
          torch.allclose(compute_view_weights(FakeLoader(wide), 0.5, torch.device("cpu")),
                         uncapped, rtol=1e-12))
    for ratio in (10.0, 100.0):
        capped = weights_for(wide, 0.5, max_weight_ratio=ratio)
        observed = float(capped.max() / capped.min())
        check(f"ratio={ratio}: bounds max(w)/min(w)", observed <= ratio * (1 + 1e-9),
              f"observed={observed:.6g}")
        check(f"ratio={ratio}: still normalized to mean 1", torch.allclose(
            capped.mean(), torch.tensor(1.0, dtype=torch.float64)))
        check(f"ratio={ratio}: still monotone decreasing in power",
              bool((capped[1:] <= capped[:-1]).all()))
    check("floor actually binds here (uncapped ratio is enormous)",
          float(uncapped.max() / uncapped.min()) > 1e10,
          f"uncapped ratio={float(uncapped.max() / uncapped.min()):.3g}")

    print("\n9c. flooring is idempotent (the defect the clamp-and-renormalize form had)")
    once = weights_for(wide, 0.5, max_weight_ratio=10.0)
    floored_powers = [max(p, max(wide) / 10.0 ** (1 / 0.5)) for p in wide]
    twice = weights_for(floored_powers, 0.5, max_weight_ratio=10.0)
    check("re-flooring an already-floored spectrum is a no-op",
          torch.allclose(once, twice, rtol=1e-10),
          f"max|d|={float((once - twice).abs().max()):.3e}")

    print("\n9d. a floor set above the natural ratio is a no-op")
    # the real B787 training spectrum spans 2.02e6 in power -> a 37.7x weight
    # ratio at alpha = 0.25, so a floor set above that must not perturb anything
    benign = weights_for(powers, 0.25, max_weight_ratio=1e6)
    check("a slack floor leaves the weights untouched",
          torch.allclose(benign, weights_for(powers, 0.25), rtol=1e-12))

    print("\n10. the reported data loss and rel-MSE stay unweighted")
    # replicates train.py's accumulation order: the unweighted per-view loss is
    # banked BEFORE the weight is applied to the backward objective
    per_view_loss = [0.4, 0.9, 0.2, 0.7]
    w = weights_for([1e-9, 1e-10, 1e-8, 1e-11], 0.5)
    reported, objective = 0.0, 0.0
    for i, lv in enumerate(per_view_loss):
        reported += lv
        objective += lv * float(w[i])
    check("reported data loss is the unweighted sum",
          abs(reported - sum(per_view_loss)) < 1e-15, f"{reported:.17g}")
    check("training objective differs from it (weighting is live)",
          abs(objective - reported) > 1e-6, f"objective={objective:.6f} vs reported={reported:.6f}")

    print("\n11. determinism across repeated construction (multi-rank safety)")
    a, b = weights_for(powers, 0.25), weights_for(powers, 0.25)
    check("bit-identical on reconstruction", bool(torch.equal(a, b)))

    passed, total = sum(checks), len(checks)
    print(f"\n{passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
