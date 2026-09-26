#!/usr/bin/env python
"""Convert raw Ansys HFSS .frtm B787 RCS simulation output
(data/ansysRCS/B787_50m/zone1-5) to per-viewpoint CSVs in RIFT's training
format (see rift/dataset.py: freq, dphi, dtheta, then alternating
magnitude/phase columns per Tx/Rx pair).

Promotes notebooks/data_prep/reformulate_data_06201.ipynb to a script and
fixes it for the current data layout: that notebook hardcoded a single-zone
input path (`data/AEDT_Multi_Scene/zone5/ParametricSetup1_ts_10_9_2025_...`)
that does not exist anywhere on this filesystem (checked project and home
dirs), only converted one zone at a time, and wrote to a single merged
output directory instead of the per-zone CSV training layout
(data/AEDT_B787_Sample/zone{N}_CSV, one directory per zone). This script
loops over all 5 real zone directories and writes to the expected per-zone
paths.

Also avoids the notebook's approach of renaming the raw .frtm files in
place (inserting "_Scattered_" so read_frtm.get_results_files' naming
convention would match) -- this script instead matches the files' ACTUAL
naming pattern (ChirpIQ_KNG4JP_Sweep_DV<id>.frtm) directly, so the raw HFSS
export files are never modified.

IMPORTANT -- discovered by inspecting the raw data directly, not documented
anywhere before this: the B787 simulation used 8.5-11.5 GHz (not
rift/config.py's hardcoded 95-105 GHz) and a 16 Tx x 15 Rx array, 240
channels (not the 16x16=256 rift/config.py assumes). Confirmed consistent
across zone1, zone2, zone3, zone5 (spot-checked). Converting produces
correct CSVs regardless, but training on this data with rift/config.py's
current values UNCHANGED would use the wrong wavelength and the wrong
Rx array geometry -- do not point train.py at this data without addressing
that first.

Usage:
    python scripts/convert_frtm_to_csv.py --limit 3   # validate on a few files first
    python scripts/convert_frtm_to_csv.py              # full conversion, all zones
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from external.read_frtm.read_frtm import read_frtm


def list_frtm_entries(setup_dir):
    """Read index.csv in setup_dir and return [(dphi, dtheta, var_id, frtm_path), ...]
    using the files' actual naming convention (no renaming).
    """
    index_files = glob.glob(os.path.join(setup_dir, "*.csv"))
    if len(index_files) != 1:
        raise FileNotFoundError(f"expected exactly 1 index.csv in {setup_dir}, found {len(index_files)}")
    index_path = index_files[0]

    entries = []
    with open(index_path, "r") as f:
        for row in csv.DictReader(f):
            var_id = row["Var_ID"]
            dphi = float(row["dphi"])
            dtheta = float(row["dtheta"])
            frtm_path = os.path.join(setup_dir, f"ChirpIQ_KNG4JP_Sweep_DV{var_id}.frtm")
            entries.append((dphi, dtheta, var_id, frtm_path))
    return entries


def convert_one(dphi, dtheta, var_id, frtm_path, output_dir):
    reader = read_frtm(frtm_path)
    freq_sweep = np.linspace(reader.freq_start, reader.freq_stop, reader.nfreq)

    columns = {
        "Frequency": freq_sweep,
        "dphi": np.full(reader.nfreq, dphi),
        "dtheta": np.full(reader.nfreq, dtheta),
    }
    for ch in reader.channel_names:
        snapshot = reader.all_data[ch][-1, :]  # last time step (stop-and-go approximation, matches original notebook)
        columns[f"{ch}_magnitude"] = np.abs(snapshot)
        columns[f"{ch}_phase"] = np.angle(snapshot)

    df = pd.DataFrame(columns)
    out_path = os.path.join(output_dir, f"dphi_{dphi}_dtheta_{dtheta}_ID_{var_id}.csv")
    df.to_csv(out_path, index=False)
    return out_path


def convert_zone(zone_dir, output_dir, limit=None):
    subdirs = [d for d in glob.glob(os.path.join(zone_dir, "*")) if os.path.isdir(d)]
    if len(subdirs) != 1:
        raise FileNotFoundError(f"expected exactly 1 setup subdirectory under {zone_dir}, found {len(subdirs)}")
    setup_dir = subdirs[0]

    os.makedirs(output_dir, exist_ok=True)
    entries = list_frtm_entries(setup_dir)
    if limit is not None:
        entries = entries[:limit]

    written, skipped = [], []
    for dphi, dtheta, var_id, frtm_path in entries:
        if not os.path.exists(frtm_path):
            skipped.append((var_id, "frtm file missing"))
            continue
        try:
            written.append(convert_one(dphi, dtheta, var_id, frtm_path, output_dir))
        except Exception as e:
            skipped.append((var_id, str(e)))
    return written, skipped


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zones-root", default="data/ansysRCS/B787_50m")
    p.add_argument("--output-root", default="data/AEDT_B787_Sample")
    p.add_argument("--zones", nargs="+", default=["zone1", "zone2", "zone3", "zone4", "zone5"])
    p.add_argument("--limit", type=int, default=None, help="Convert only the first N files per zone (for validation)")
    args = p.parse_args()

    for zone in args.zones:
        zone_dir = os.path.join(args.zones_root, zone)
        output_dir = os.path.join(args.output_root, f"{zone}_CSV")
        written, skipped = convert_zone(zone_dir, output_dir, limit=args.limit)
        print(f"{zone}: wrote {len(written)} CSVs to {output_dir}, skipped {len(skipped)}")
        for var_id, reason in skipped:
            print(f"  skipped Var_ID={var_id}: {reason}")


if __name__ == "__main__":
    main()
