#!/usr/bin/env python
"""Scrape per-epoch training metrics out of the SLURM logs.

train.py prints the validation relative MSE immediately *before* the matching
`Epoch [n/N] Training Summary:` block, followed by the global train rel-MSE,
calibration gain and active-voxel count inside the block. Nothing writes the
relative metrics to a machine-readable file -- the per-run CSV holds raw
losses, not rel-MSE -- so the logs are the only source for the headline metric.

Runs chain across embers preemptions, so several `Report-*.out` files can carry
the same checkpoint name with overlapping epoch ranges. Files are read in mtime
order and later files win per epoch, which reconstructs the run as it actually
proceeded.

Logs for FINISHED runs are filed under `archive/slurm_reports/by_run/<run>/`
(2026-08-05, to keep the repo root readable); only in-flight runs write to the
root. DEFAULT_LOG_GLOBS covers both, so a chain that spans the archive and the
root is still reconstructed -- do not narrow it to just `Report-*.out`.

    module load anaconda3 && conda activate RIFT
    python scripts/scrape_training_logs.py                 # summary table
    python scripts/scrape_training_logs.py --json curves.json
"""
import argparse
import collections
import glob
import json
import os
import re

EPOCH_RE = re.compile(r"Epoch \[(\d+)/(\d+)\] Training Summary")
FIELDS = [
    (re.compile(r"Global relative MSE.*?:\s*([\d.]+)%"), "train", float),
    (re.compile(r"\|g\| = ([\d.eE+-]+)"), "g", float),
    (re.compile(r"Pruned:\s*(\d+)/(\d+)"), "active", int),
]
VAL_RE = re.compile(r"Validation relative MSE:\s*([\d.]+)%")


DEFAULT_LOG_GLOBS = ("Report-*.out", "archive/slurm_reports/by_run/*/Report-*.out")


def resolve_logs(log_globs):
    """Expand one glob, or an iterable of globs, to a de-duplicated file list."""
    if isinstance(log_globs, str):
        log_globs = (log_globs,)
    paths = {p for pattern in log_globs for p in glob.glob(pattern)}
    return sorted(paths, key=os.path.getmtime)


def scrape(log_glob=DEFAULT_LOG_GLOBS):
    """-> {checkpoint_name: {"epochs": [ {epoch, train, val, g, active}, ... ],
                             "args": str, "total_epochs": int}}"""
    runs = collections.defaultdict(dict)
    meta = {}
    for path in resolve_logs(log_glob):
        text = open(path, errors="ignore").read()
        m = re.search(r"--checkpoint-name\s+(\S+)", text)
        if not m:
            continue
        name = m.group(1)
        argline = re.search(r"RIFT training: (.*)", text)
        meta.setdefault(name, {"args": argline.group(1) if argline else "", "total_epochs": None})
        cur = None
        pending_val = None
        for line in text.splitlines():
            vm = VAL_RE.search(line)
            if vm:
                # evaluate() prints this line before the matching epoch summary.
                # Holding it until the next EPOCH_RE avoids the historical
                # off-by-one association and retains the final epoch's value.
                pending_val = float(vm.group(1))
                continue
            em = EPOCH_RE.match(line)
            if em:
                cur = int(em.group(1))
                runs[name][cur] = {"epoch": cur}
                if pending_val is not None:
                    runs[name][cur]["val"] = pending_val
                    pending_val = None
                meta[name]["total_epochs"] = int(em.group(2))
                continue
            if cur is None:
                continue
            for pat, key, cast in FIELDS:
                fm = pat.search(line)
                if fm:
                    runs[name][cur][key] = cast(fm.group(1))
    return {name: {"epochs": [ep[e] for e in sorted(ep)], **meta[name]}
            for name, ep in runs.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", nargs="+", default=list(DEFAULT_LOG_GLOBS),
                   help="glob(s) of SLURM logs to scrape (run from the repo root); "
                        "default covers the root and archive/slurm_reports/by_run/")
    p.add_argument("--json", help="also write the full per-epoch curves here")
    args = p.parse_args()

    runs = scrape(args.logs)
    if args.json:
        json.dump(runs, open(args.json, "w"))
        print(f"wrote {args.json}")

    print(f"{'run':40s} {'epochs':>14s} {'best val':>9s} {'@ep':>5s} "
          f"{'train':>8s} {'|g|':>10s} {'active':>9s}")
    for name in sorted(runs):
        cur = [d for d in runs[name]["epochs"] if "val" in d]
        if not cur:
            continue
        best = min(cur, key=lambda d: d["val"])
        reached, total = max(d["epoch"] for d in cur), runs[name]["total_epochs"]
        print(f"{name:40s} {f'{reached}/{total}':>14s} {best['val']:8.2f}% "
              f"{best['epoch']:5d} {best.get('train', float('nan')):7.2f}% "
              f"{best.get('g', float('nan')):10.3e} {str(best.get('active', '-')):>9s}")


if __name__ == "__main__":
    main()
