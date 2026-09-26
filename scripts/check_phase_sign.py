#!/usr/bin/env python
"""Determine a dataset's propagation-phase sign convention in one command.

The sign of exp(phase_sign * i * k * R) is a PER-SIMULATOR convention:
this project's AEDT/HFSS export pipeline (sphere + B787) is +1 (confirmed
2026-07-03), and Ansys AVXcelerate is expected to be +1 as well -- but verify
every new data source with this script before training on it. Fitting with
the wrong sign mirrors the scene differently at every viewpoint, so no single
scene can fit multi-viewpoint data of a non-spherical target.

METHOD -- range-profile causality (alias-immune). Naive approaches (fitting a
scatterer under both signs, comparing wrapped phase slopes) FAIL on uniformly
frequency-gridded data: (sign, range R) and (-sign, m*c/(2*df) - R) produce
identical single-viewpoint data up to a constant phase, and simulation setups
routinely place the target near the alias-symmetric range (both this
project's datasets do). Instead this script uses causality: nothing can
scatter EARLIER than the target's first illuminated surface, so the true
range profile has a sharp leading edge and an extended tail BEHIND the main
peak (target body, multibounce, creeping waves). np.fft.ifft reads
e^{-i2pi*f*tau} data as a peak at RT = c*tau, so:
  - energy tail AFTER the as-read peak  -> data is e^{-i...} -> phase_sign -1
  - energy tail BEFORE the as-read peak -> data is e^{+i...} -> phase_sign +1

Usage:
    python scripts/check_phase_sign.py --data-dir data/AEDT_B787_Sample/zone1_CSV
    python scripts/check_phase_sign.py --data-dir <avx_csvs> --num-viewpoints 20
"""
import argparse
import glob
import os

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, help="Directory of per-viewpoint CSVs (train.py format)")
    p.add_argument("--num-viewpoints", type=int, default=10,
                   help="Viewpoints to test, spread evenly over the directory")
    p.add_argument("--tail-meters", type=float, default=None,
                   help="Tail window (RT meters) each side of the peak; default = 1/3 of the "
                        "unambiguous window")
    p.add_argument("--guard-meters", type=float, default=0.5,
                   help="Guard band around the main peak excluded from both windows")
    return p.parse_args()


def analyze(path, tail_meters, guard_meters):
    d = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=float)
    freqs = d[:, 0]
    df = np.diff(freqs).mean()
    if np.diff(freqs).std() > 0.01 * df:
        raise ValueError(f"{path}: frequency rows are not uniformly spaced; this test assumes a uniform grid")
    S = d[:, 3::2] * np.exp(1j * d[:, 4::2])
    prof = np.abs(np.fft.ifft(S, axis=0)).mean(axis=1)   # channel-averaged range profile
    N = len(prof)
    bin_m = 299792458.0 / (N * df)                        # RT meters per bin
    window = N * bin_m
    tail = tail_meters if tail_meters is not None else window / 3.0
    peak = int(np.argmax(prof))
    nb_tail = max(1, int(tail / bin_m))
    nb_guard = max(1, int(guard_meters / bin_m))
    after = [(peak + nb_guard + j) % N for j in range(nb_tail)]
    before = [(peak - nb_guard - j) % N for j in range(nb_tail)]
    E_after = float((prof[after] ** 2).sum())
    E_before = float((prof[before] ** 2).sum())
    return 10 * np.log10(E_after / E_before), peak * bin_m, window


def main():
    args = parse_args()
    files = sorted(glob.glob(os.path.join(args.data_dir, "*.csv")))
    if not files:
        raise SystemExit(f"no CSVs in {args.data_dir}")
    step = max(1, len(files) // args.num_viewpoints)
    picked = files[::step][:args.num_viewpoints]

    ratios = []
    for f in picked:
        r, peak_m, window = analyze(f, args.tail_meters, args.guard_meters)
        ratios.append(r)
        lean = "-1" if r > 0 else "+1"
        print(f"{os.path.basename(f)[:55]:55s}  peak {peak_m:7.2f}m / {window:.0f}m window, "
              f"tail asymmetry {r:+6.1f} dB -> {lean}")

    mean = float(np.mean(ratios))
    votes_minus = sum(1 for r in ratios if r > 0)
    winner = "-1" if mean > 0 else "+1"
    strength = "clear" if abs(mean) > 2.0 else "WEAK -- inspect the range profiles manually"
    print(f"\nmean asymmetry {mean:+.1f} dB, votes: -1 x{votes_minus} / +1 x{len(ratios) - votes_minus}")
    print(f"=> use --phase-sign {winner} for this dataset ({strength})")


if __name__ == "__main__":
    main()
