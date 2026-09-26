#!/usr/bin/env python
"""Extract per-step traces from a ``train_sas_pvc.py`` log (docs/RIFT_SAS_Train.md, R1 pilot).

Pairs each ``step N/T loss=... rel_mse=... grad=...`` line with the following
``sonar_diagnostics`` line and writes a CSV with the one-ping loss/rel-MSE (never
quoted as TRAIN), gradient norm, gain, raw correlation, transmittance
(``T_all_min``/``T_all_mean``) and the Lambertian gate (``lambert_positive``), so
a Lambertian-gate failure can be told from a learning-rate failure. Also prints
the validation lines and a windowed summary.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

STEP_RE = re.compile(r"^step (\d+)/\d+ loss=(\S+) rel_mse=(\S+) grad=(\S+) rays=(\d+)")
DIAG_RE = re.compile(r"^sonar_diagnostics (.*)$")
VAL_RE = re.compile(r"^validation step=(\d+) rel_mse=(\S+) views=(\d+)")
FIELDS = ("raw_corr_abs", "gain_abs", "T_all_min", "T_all_mean", "T_all_lt_1e3", "lambert_positive")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--windows", type=int, default=5)
    options = parser.parse_args(argv)
    rows, validation, pending = [], [], None
    for line in Path(options.log).read_text(errors="replace").splitlines():
        match = STEP_RE.match(line)
        if match:
            pending = {"step": int(match[1]), "loss": float(match[2]), "one_ping_rel_mse": float(match[3]),
                       "grad_norm": float(match[4])}
            continue
        match = DIAG_RE.match(line)
        if match and pending is not None:
            values = dict(item.split("=", 1) for item in match[1].split())
            pending.update({key: float(values[key]) for key in FIELDS if key in values})
            rows.append(pending)
            pending = None
            continue
        match = VAL_RE.match(line)
        if match:
            validation.append((int(match[1]), float(match[2]), int(match[3])))
    if not rows:
        print("no step/sonar_diagnostics pairs found")
        return 1
    with open(options.csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} logged steps -> {options.csv}")
    for step, rel, views in validation:
        print(f"VAL{views} step {step}: rel-MSE {rel:.4f}")
    chunks = np.array_split(np.arange(len(rows)), min(options.windows, len(rows)))
    print("window  steps          median one-ping rel-MSE  median grad  lambert_positive  T_all_min  T_all_mean  gain_abs")
    for chunk in chunks:
        part = [rows[i] for i in chunk]
        med = lambda key: float(np.median([r.get(key, np.nan) for r in part]))  # noqa: E731
        print(f"  {part[0]['step']:>6d}-{part[-1]['step']:<6d}  {med('one_ping_rel_mse'):20.4f}  {med('grad_norm'):11.3e}  "
              f"{med('lambert_positive'):16.3f}  {med('T_all_min'):9.3f}  {med('T_all_mean'):10.3f}  {med('gain_abs'):.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
