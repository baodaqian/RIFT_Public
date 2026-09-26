"""Gate for prune()'s threshold modes, min-active floor and target ramp (2026-08-03).

Run: PYTHONPATH=. python scripts/validate_prune_modes.py

Round 4 died of the legacy "relmax" prune rule, which recomputes its
reference (the max) at every check against an ever-more-concentrated
distribution and therefore has no fixed point but the empty scene. This gate
proves the two replacements do have one:

  * mass   -- discards the requested fraction of the scene's total angular
              energy per check, and no more.
  * target -- converges to --prune-target-active and HOLDS there under
              repeated application.

Block 4 is the regression that would have caught the R4 failure: it replays
the actual R4 schedule (--prune-every 10, threshold 0.01) on a peaked
distribution and asserts relmax runs away while mass/target do not.

The sharded twin is covered by scripts/validate_distributed_scene.py, which
must be run at world_size >= 2.
"""
import torch

from rift.sparse_scene import (SHVoxelGridScene, _global_mass_below,
                               _threshold_for_mass, prune_threshold_and_report)

dev = "cpu"
torch.manual_seed(0)


def make_scene(granularity=16, max_degree=2, peaked=True):
    """A scene whose energy is concentrated on a thin shell, like a real PEC
    reconstruction -- the peaked case is where relmax bites hardest."""
    s = SHVoxelGridScene(granularity, 1.5, dev, max_degree=max_degree,
                         init_degree=max_degree)
    n = granularity
    idx = torch.arange(n, dtype=torch.float64)
    c = (n - 1) / 2.0
    r = torch.sqrt(((idx - c) ** 2).reshape(-1, 1, 1)
                   + ((idx - c) ** 2).reshape(1, -1, 1)
                   + ((idx - c) ** 2).reshape(1, 1, -1))
    shell = torch.exp(-((r - 0.35 * n) ** 2) / (0.7 if peaked else 8.0))
    with torch.no_grad():
        amp = (shell + 1e-4).to(s.w_re.dtype).unsqueeze(-1)
        s.w_re.copy_(amp * torch.rand_like(s.w_re))
        s.w_im.copy_(amp * torch.rand_like(s.w_im))
    return s


def energy(scene):
    unlocked = (scene.basis_degree.view(1, 1, 1, -1) <= scene.order.unsqueeze(-1))
    sq = (scene.w_re ** 2 + scene.w_im ** 2) * unlocked.to(scene.w_re.dtype)
    return sq.sum(dim=-1).sqrt()


# --- 1. mass mode spends exactly the requested energy budget ---------------
print("--- mass mode: energy actually discarded vs requested")
for spread in ("peaked", "broad"):
    vals = energy(make_scene(peaked=(spread == "peaked")))
    vals = vals[vals > 0].reshape(-1)
    total = float((vals ** 2).sum())
    for req in (0.05, 0.01, 0.001):
        t = _threshold_for_mass(vals, vals.max(), req, vals.device, False)
        spent = float((vals[vals < t] ** 2).sum()) / total
        print(f"  {spread:8s} requested={req:<8} discarded={spent:.6f}  "
              f"kept {float((vals >= t).float().mean()):.4f} of entries")
        assert spent <= req * 1.05 + 1e-9, f"overspent: {spent} > {req}"
        assert spent >= req * 0.5, f"underspent badly: {spent} vs {req}"

# --- 2. _global_mass_below is exact ---------------------------------------
print("--- mass report exactness")
vals = torch.exp(torch.randn(50000, dtype=torch.float64))
ts = torch.tensor([0.1, 0.5, 1.0, 3.0], dtype=torch.float64)
got = _global_mass_below(vals, ts, vals.device, False)
ref = torch.stack([(vals[vals < t] ** 2).sum() / (vals ** 2).sum() for t in ts])
print("  max abs err:", float((got - ref).abs().max()))
assert torch.allclose(got, ref, atol=1e-12), (got, ref)

# --- 3. target mode hits the count and HOLDS it ---------------------------
print("--- target mode: converges and stays (the fixed point relmax lacks)")
for target in (2000, 500, 50):
    s = make_scene()
    for check in range(8):
        n_active, n_total, _ = s.prune(mode="target", target_active=target)
    print(f"  target={target:<6} after 8 repeated prunes: {n_active}/{n_total} active")
    assert abs(n_active - target) <= max(2, int(0.02 * target)), \
        f"target {target}, got {n_active}"

# --- 4. THE R4 REGRESSION: relmax runs away, mass/target do not -----------
#
# Pruning ALONE is idempotent for relmax -- the max survives its own cut, so
# re-applying the same threshold to frozen weights removes nothing more
# (verified in block 4b). The R4 runaway is the composition prune-then-RETRAIN:
# each prune is followed by ~10 epochs in which the optimizer pushes the
# remaining energy onto the surviving scatterers, so the distribution the NEXT
# check sees is more peaked than the one before it, and the same nominal
# 1%-of-max cut bites deeper. That is the loop this block reproduces.
#
# `_reconcentrate` is the stand-in for those 10 epochs: it sharpens the
# surviving weights toward the peak (w *= (mag/mag.max())**0.5), which is the
# qualitative thing training does to a scene that has just lost its weakest
# scatterers. Every mode is evolved by the SAME map, so the comparison is fair
# -- what differs is only how each rule responds to a sharpening distribution.
@torch.no_grad()
def _reconcentrate(scene, gamma=0.5):
    mag = energy(scene)
    scale = (mag / mag.max().clamp_min(1e-30)) ** gamma
    scene.w_re.mul_(scale.unsqueeze(-1))
    scene.w_im.mul_(scale.unsqueeze(-1))


print("--- R4 replay: 8 prune+retrain rounds at the settings that killed Round 4")
n_total = None
traces, spends = {}, {}
for mode, kw in (("relmax", dict(threshold_fraction=0.01)),
                 ("mass", dict(threshold_fraction=0.01)),
                 ("target", dict(target_active=1000))):
    s = make_scene()
    trace, spend = [], []
    for check in range(8):
        before = float((energy(s) ** 2).sum())
        n_active, n_total, _ = s.prune(mode=mode, **kw)
        after = float((energy(s) ** 2).sum())
        trace.append(n_active)
        spend.append(1.0 - after / before)
        _reconcentrate(s)
    traces[mode], spends[mode] = trace, spend
    print(f"  {mode:7s} active " + " -> ".join(str(x) for x in trace))
    print(f"          energy discarded per check: "
          + ", ".join(f"{x:.2%}" for x in spend))

# What each rule actually guarantees, and what this block can and cannot show:
#
#   relmax -- NOTHING is bounded. Its cut is a fraction of a moving reference,
#             so every retrain step lets it bite again: the COUNT collapses
#             monotonically with no fixed point. This is the R4 failure.
#   mass   -- a HARD per-check bound on discarded ENERGY. Its count can still
#             fall (a sharpening scene really does concentrate its energy into
#             fewer voxels, and dropping voxels that carry ~nothing is the
#             POINT), but it can never spend more signal than asked. Pair with
#             --prune-min-active to bound the count as well.
#   target -- a hard fixed point in COUNT.
#
# READ THE ENERGY NUMBERS CAREFULLY. Under this proxy relmax wipes out ~90% of
# the voxels while discarding well under 0.1% of the energy -- i.e. the energy
# metric does NOT detect the damage relmax does. That is a statement about the
# metric, not a defense of relmax: a coherent sum needs its small terms, and
# the project has already measured the same thing on real data (the good B787
# fit keeps 62% of its energy OFF the airframe and breaks when those DOF are
# removed -- see the --extent 0.06 A/B). The real evidence that relmax is
# lethal is the R4 training curve, where train rel-MSE jumps at every prune
# check (1.1% -> 4.1% -> 8.8% -> 20% -> 39% -> 85% -> 100%). A synthetic scene
# with no data term cannot reproduce that, so this gate deliberately asserts
# only the MECHANICS. Which mode actually trains better is R5's job.
assert traces["relmax"][-1] < 0.2 * traces["relmax"][0], \
    f"relmax no longer runs away: {traces['relmax']}"
assert all(b <= a for a, b in zip(traces["relmax"], traces["relmax"][1:])), \
    f"relmax count should decay monotonically: {traces['relmax']}"
for x in spends["mass"]:
    assert x <= 0.01 * 1.05 + 1e-9, f"mass overspent its budget: {spends['mass']}"
assert traces["target"][-3:] == [1000, 1000, 1000], traces["target"]
print(f"  => relmax kept {traces['relmax'][-1]}/{traces['relmax'][0]} of its voxels "
      f"while discarding only {max(spends['relmax']):.3%} of the energy "
      f"-- energy is NOT the guard rail; count and fit are.")

# --- 4b. prune alone (no retrain) is idempotent, even for relmax ----------
print("--- prune alone is idempotent (isolates WHICH step ratchets)")
s = make_scene()
alone = []
for check in range(5):
    n_active, _, _ = s.prune(threshold_fraction=0.01, mode="relmax")
    alone.append(n_active)
print("  relmax, frozen weights: " + " -> ".join(str(x) for x in alone))
assert len(set(alone)) == 1, \
    f"relmax should be idempotent without a retrain step, got {alone}"

# --- 5. min-active floor is honored by EVERY mode -------------------------
print("--- min-active floor")
for mode, kw in (("relmax", dict(threshold_fraction=0.5)),
                 ("mass", dict(threshold_fraction=0.9)),
                 ("target", dict(target_active=10))):
    s = make_scene()
    for check in range(5):
        n_active, n_total, _ = s.prune(mode=mode, min_active=1500, **kw)
    print(f"  {mode:7s} floor=1500 -> {n_active} active after 5 aggressive prunes")
    assert n_active >= 1500, f"{mode} broke the floor: {n_active}"

# --- 6. relmax is bit-identical to the pre-2026-08-03 rule ----------------
print("--- relmax reproduces the legacy rule exactly")
s = make_scene()
mag = energy(s)
legacy = s.active_mask & (mag >= 0.01 * mag.max())
n_active, _, report = s.prune(threshold_fraction=0.01, mode="relmax")
assert int(legacy.sum()) == n_active, (int(legacy.sum()), n_active)
assert torch.equal(legacy, s.active_mask)
print(f"  legacy mask == new mask ({n_active} active)")
print(f"  report: {report}")

# --- 7. the 3-tuple train.py expects, on every scene class ----------------
print("--- return signature")
from rift.sparse_scene import AdaptivePointSHScene, VoxelGridScene
g = VoxelGridScene(8, 1.5, dev)
out = g.prune(threshold_fraction=0.01, mode="mass")
assert len(out) == 3, out
p = AdaptivePointSHScene(torch.rand(500, 3) * 2 - 1, 0.05, dev,
                         max_degree=1, init_degree=1, init_scale=0.1)
out = p.prune(threshold_fraction=0.01, mode="mass")
assert len(out) == 3, out
print("  VoxelGridScene / AdaptivePointSHScene / SHVoxelGridScene all return "
      "(n_active, n_total, report)")

# --- 8. the target ramp itself --------------------------------------------
print("--- cubic target ramp")
import sys
sys.path.insert(0, ".")
from train import _prune_target_for_epoch
n = 110592
pts = [(e, _prune_target_for_epoch(e, "target", 4000, 30, 120, n))
       for e in (30, 45, 60, 90, 120, 150)]
print("  " + ", ".join(f"ep{e}->{v}" for e, v in pts))
assert pts[0][1] == n, pts[0]
assert pts[-1][1] == 4000 and pts[-2][1] == 4000, pts[-2:]
assert all(pts[i][1] >= pts[i + 1][1] for i in range(len(pts) - 1)), "not monotone"
assert _prune_target_for_epoch(50, "relmax", 0, 30, 120, n) is None

print("\nALL PRUNE-MODE CHECKS PASSED")
