"""Pick the least-contended PACE CPU-only partition that can run a job needing
N cores on one node right now.

Companion to pick_gpu_partition.py, same idea applied to the CPU pools: rank
by nodes-that-can-start-it-NOW, then by pending queue depth. embers QOS has
UsageFactor=0.0 (verified via `sacctmgr show qos embers`), so there is no
billing difference between these partitions -- ranking on availability alone
costs nothing.

Only DISJOINT node pools are listed. cpu-medium and cpu-large were checked
against cpu-small's node list (2026-07-26) and found to be 100% subsets --
same physical nodes, different priority tier, not separate capacity -- so
including them would let the picker "choose" the same hardware twice under a
different name. cpu-amd/cpu-gnr/cpu-sas are confirmed disjoint from cpu-small
by node-name prefix and by comm -12 producing zero overlap.

Usage:
    python scripts/pick_cpu_partition.py                # default: 24 cores (Resolve_SB's baked ask)
    python scripts/pick_cpu_partition.py --cpus 16
"""
from __future__ import annotations

import argparse
import subprocess
from collections import defaultdict

# partition -> (label, disjoint from cpu-small: yes for all of these by construction)
PARTITIONS = {
    "cpu-amd":   "AMD EPYC",
    "cpu-gnr":   "Intel Granite Rapids",
    "cpu-sas":   "SAS pool",
    "cpu-small": "General pool (default)",
}

DEAD_STATES = ("down", "drain", "drng", "resv", "unk", "fail", "maint", "boot")


def run(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def cpu_free(field):
    """CPUsState 'A/I/O/T' (alloc/idle/other/total) -> idle count."""
    parts = field.split("/")
    return int(parts[1]) if len(parts) == 4 else 0


def collect(partitions):
    out = run("sinfo -h -N -O NodeList:24,Partition:20,StateCompact:12,CPUsState:20 -p " +
              ",".join(partitions))
    parts = defaultdict(lambda: {"nodes": 0, "free_per_node": []})
    seen = set()
    for line in out.splitlines():
        f = line.split()
        if len(f) < 4:
            continue
        node, part, state, cpus = f[0], f[1], f[2], f[3]
        part = part.rstrip("*")
        if part not in partitions or (node, part) in seen:
            continue
        seen.add((node, part))
        p = parts[part]
        p["nodes"] += 1
        if any(state.startswith(d) for d in DEAD_STATES):
            continue
        p["free_per_node"].append(cpu_free(cpus))

    pend = defaultdict(int)
    for line in run("squeue -h -t PENDING -O Partition:40").splitlines():
        for part in line.strip().split(","):
            part = part.strip()
            if part in partitions:
                pend[part] += 1
    return parts, pend


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cpus", type=int, default=24,
                    help="cores needed on one node (default 24, Resolve_SB's --cpus-per-task)")
    args = ap.parse_args()

    parts, pend = collect(set(PARTITIONS))

    rows = []
    for part, label in PARTITIONS.items():
        p = parts.get(part)
        if not p or p["nodes"] == 0:
            continue
        ready = sum(1 for fr in p["free_per_node"] if fr >= args.cpus)
        rows.append((part, label, p["nodes"], ready))

    rows.sort(key=lambda r: (-r[3], pend.get(r[0], 0)))

    print(f"Target: {args.cpus} core(s) on one node\n")
    print(f"{'partition':<14}{'label':<26}{'nodes':>7}{'ready':>7}{'pending':>9}")
    print("-" * 63)
    for part, label, nodes, ready in rows:
        print(f"{part:<14}{label:<26}{nodes:>7}{ready:>7}{pend.get(part, 0):>9}")

    if not rows:
        print("\nNo eligible CPU partition found.")
        return

    part, label, nodes, ready = rows[0]
    print(f"\n--> {part} ({label}): {ready} node(s) can start it now, {pend.get(part, 0)} jobs queued.")
    print(f"    sbatch --partition={part} --cpus-per-task={args.cpus} <script>")
    if ready == 0:
        print("    (nothing is startable right now anywhere -- this is a queue wait,"
              " so the ranking fell through to pending depth)")


if __name__ == "__main__":
    main()
