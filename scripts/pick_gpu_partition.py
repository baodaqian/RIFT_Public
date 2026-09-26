"""Pick the least-contended PACE partition that can actually run a RIFT job,
sizing the GPU COUNT so every option delivers the same throughput.

DEFAULT SPEC (Daqian, 2026-07-26): a **2x H200-equivalent** allocation, taken
on whichever eligible partition can start it soonest. That is 2 GPUs on
h100/h200, or "a few" L40S / RTX-PRO -- the script scales the count by each
card's slowdown factor k so the choice never costs speed, only GPUs. A small
allocation also backfills into node fragments, whereas a whole-node request on
preemptible embers is both rare and the first thing evicted.

**v100 and rtx6000 are dropped from speed-demand runs** -- not on memory (both
fit every RIFT config) but on throughput and GPUs-per-node (2 and 4); they are
also the two most contended partitions on the machine. `--include-slow` puts
them back for probes and one-off inference.

Memory is never the discriminator: peak VRAM is O(point_chunk*pair_chunk), not
O(n_points) -- ~2 GB/rank for the g96 B787 scene, ~3.9 GB even unsharded (see
the 2026-07-26 handoff section). So this ranks on availability alone, having
already equalized throughput via k.

Usage:
    python scripts/pick_gpu_partition.py                  # THE DEFAULT SPEC (2x H200-equivalent)
    python scripts/pick_gpu_partition.py --gpu-equiv 8    # the original 8xH200-class ask
    python scripts/pick_gpu_partition.py --gpus 1         # exactly 1 GPU, no scaling (D0-style)
    python scripts/pick_gpu_partition.py --include-slow   # allow v100/rtx6000

The scene-sharded launcher is single-node (`torchrun --nnodes=1`), so a node
must have the required GPUs FREE ON ITSELF -- that is what "ready" counts.
"""
from __future__ import annotations

import argparse
import math
import re
import subprocess
from collections import defaultdict

# card -> (label, VRAM GB, nominal fp64 TFLOPS, k, k_measured, speed_eligible)
#
# k = slowdown per GPU vs one H200 on THIS workload. fp64 is the right basis:
# the range operator's phase path is float64 by mandate (CLAUDE.md), and fp64
# is exactly where the consumer parts are cut down.
#
# !! The k values below are PREDICTIONS, not measurements. On the H200 anchor
# the kernel achieves only ~0.2 TFLOPS fp64 (0.5% of peak) and ~300 GB/s (6%
# of peak), so it is bound by neither -- most likely by fp64 atomics in
# index_add_ -- which is why the 24x fp64 spec gap to L40S is expected to
# translate into a far smaller real gap. Measure with
# scripts/bench_forward_operator.py and set k_measured=True here.
CARDS = {
    "h200":                   ("H200 SXM",                141, 34.0, 1.0, False, True),
    "h100":                   ("H100 SXM",                 80, 34.0, 1.2, False, True),
    "a100":                   ("A100-80GB",                80,  9.7, 2.0, False, True),
    "rtx_pro_6000_blackwell": ("RTX PRO 6000 Blackwell",   96,  1.9, 2.5, False, True),
    "l40s":                   ("L40S",                     48,  1.4, 3.0, False, True),
    "rtx_6000":               ("Quadro RTX 6000 (Turing)", 24,  0.5, 7.0, False, False),
    "v100":                   ("V100-16GB",                16,  7.8, 3.0, False, False),
}

DEAD_STATES = ("down", "drain", "drng", "resv", "unk", "fail", "maint", "boot")


def run(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def gpu_count(field):
    """'gpu:l40s:8(S:0-3)' / 'gpu:l40s:0(IDX:N/A)' -> ('l40s', n)."""
    m = re.search(r"gpu:([a-z_0-9]+):(\d+)", field)
    return (m.group(1), int(m.group(2))) if m else (None, 0)


def collect():
    """Per-partition GPU inventory. Field widths must exceed the longest gres
    string ('gpu:rtx_pro_6000_blackwell:8(S:0-3)') or sinfo silently truncates
    it and every count comes out wrong."""
    out = run("sinfo -h -N -O NodeList:24,Partition:26,StateCompact:12,Gres:44,GresUsed:44")
    parts = defaultdict(lambda: {"card": None, "per_node": 0, "nodes": 0,
                                 "free": 0, "free_per_node": []})
    seen = set()
    for line in out.splitlines():
        f = line.split()
        if len(f) < 5:
            continue
        node, part, state, gres, used = f[0], f[1], f[2], f[3], f[4]
        if part.startswith("interactive"):
            continue          # OnDemand sessions, not batch training
        card, total = gpu_count(gres)
        if card is None or total == 0 or card not in CARDS or (node, part) in seen:
            continue
        seen.add((node, part))
        p = parts[part]
        p["card"] = card
        p["per_node"] = max(p["per_node"], total)
        p["nodes"] += 1
        if any(state.startswith(d) for d in DEAD_STATES):
            continue          # capacity exists but is unreachable
        free = total - gpu_count(used)[1]
        p["free"] += free
        p["free_per_node"].append(free)

    pend = defaultdict(int)
    for line in run("squeue -h -t PENDING -O Partition:40").splitlines():
        for part in line.strip().split(","):
            if part.strip():
                pend[part.strip()] += 1
    return parts, pend


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu-equiv", type=float, default=2.0,
                    help="throughput target in H200-GPU-equivalents (default 2 = THE DEFAULT SPEC). "
                         "Each card's count is scaled by its k so all options run at the same speed.")
    ap.add_argument("--gpus", type=int, default=None,
                    help="exact GPU count on every partition, bypassing the k scaling "
                         "(use for single-GPU jobs where throughput parity is not the point)")
    ap.add_argument("--include-slow", action="store_true",
                    help="also consider v100/rtx6000 (excluded by default for speed-demand runs)")
    args = ap.parse_args()

    parts, pend = collect()

    rows = []
    for part, p in parts.items():
        label, vram, fp64, k, k_meas, fast = CARDS[p["card"]]
        if not fast and not args.include_slow:
            continue
        need = args.gpus if args.gpus else max(1, math.ceil(args.gpu_equiv * k))
        if p["per_node"] < need:
            continue                       # cannot be satisfied on one node
        ready = sum(1 for fr in p["free_per_node"] if fr >= need)
        rows.append((part, p, label, vram, k, k_meas, need, ready))

    # Startable NOW first -- that is the entire point of admitting L40S -- then
    # queue depth, then the option that burns fewest GPUs for the same speed.
    rows.sort(key=lambda r: (-r[7], pend.get(r[0], 0), r[6]))

    target = (f"{args.gpus} GPU(s) exactly" if args.gpus
              else f"{args.gpu_equiv:g}x H200-equivalent throughput")
    excl = "" if args.include_slow else "   (v100/rtx6000 excluded: speed-demand policy)"
    print(f"Target: {target}{excl}\n")
    print(f"{'partition':<24}{'card':<26}{'VRAM':>6}{'k':>6}{'GPUs':>6}"
          f"{'ready':>7}{'freeGPU':>9}{'pending':>9}")
    print("-" * 93)
    for part, p, label, vram, k, k_meas, need, ready in rows:
        kstr = f"{k:.1f}" + ("" if k_meas else "?")
        print(f"{part:<24}{label:<26}{vram:>4}GB{kstr:>6}{need:>6}"
              f"{ready:>7}{p['free']:>9}{pend.get(part, 0):>9}")

    if not rows:
        print("\nNo eligible partition can host this on a single node.")
        return

    part, p, label, vram, k, k_meas, need, ready = rows[0]
    script = "slurm/train_multigpu.sbatch" if need > 1 else "slurm/train.sbatch"
    print(f"\n--> {part} ({label}) x{need}: "
          f"{ready} node(s) can start it now, {pend.get(part, 0)} jobs queued.")
    print(f"    sbatch --gres=gpu:{p['card']}:{need} {script} <train.py flags>")
    if ready == 0:
        print("    (nothing is startable right now anywhere -- this is a queue wait,"
              " so the ranking fell through to pending depth)")
    if not k_meas:
        print(f"\n    NOTE: k={k:.1f} for this card is a PREDICTION (marked '?' above), so the GPU"
              f"\n    count above is only approximately throughput-matched. Confirm with:"
              f"\n        python scripts/bench_forward_operator.py"
              f"\n    on one of its GPUs (~1 min), then set k_measured=True in this script's CARDS.")
    print("\nReminders: drop any -C gpu-h200 constraint; embers has an 8 h wall and is"
          "\npreemptible, so chain with --resume (train.py checkpoints every epoch and"
          "\nreshards on load, so a preemption costs at most one epoch).")


if __name__ == "__main__":
    main()
