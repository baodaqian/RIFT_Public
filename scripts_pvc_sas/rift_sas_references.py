#!/usr/bin/env python
"""Neighbour-ring reference predictors for the AirSAS elevation-interpolation split.

Every VAL ring r (residue 7 mod 10) lies between TRAIN rings r-1 and r+1 at the
same 360 azimuths. Three reference predictors of ring r use only those
neighbours: copy of r-1, copy of r+1, and their average. They are
references to read beside a model, not lower bounds.

Metric conventions follow ``train_sas.evaluate`` (complex rel-MSE over pings and
all bins, one global complex gain):
  * ``g=1``: the neighbour rows unscaled;
  * ``g_train``: one global gain per predictor fitted on the TRAIN analog (each
    TRAIN ring whose two neighbours are TRAIN, predicted from them) and applied
    unchanged to VAL: the convention comparable to RIFT's TRAIN-fitted gain;
  * ``g_val`` and per-ping gains: report-only oracles, never compared.
The band split (FFT over the cached bins, fs from the manifest) uses the same
gains. Pooled scores are energy-weighted and the mean power falls about 40x
from the lowest to the highest rings, so a per-ring table (full band and
12.5-27.5 kHz, g = 1 and g_train, mean |y|^2) is printed for the average
predictor as a diagnostic beside the pooled numbers. Reserved-test rows are
never read (asserted).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BANDS_KHZ = ((-50.0, 0.0), (0.0, 12.5), (12.5, 27.5), (27.5, 50.0001))
IN_BAND = "12.5..27.5"


def band_masks(num_bins, sample_rate_hz):
    frequency_khz = np.fft.fftfreq(num_bins, 1.0 / sample_rate_hz) / 1e3
    return {f"{lo:g}..{min(hi, 50):g}": (frequency_khz >= lo) & (frequency_khz < hi) for lo, hi in BANDS_KHZ}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache")
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    root = Path(args.cache)
    manifest = json.loads((root / "manifest.json").read_text())
    geometry = np.load(root / "geometry.npz")
    weights = np.load(root / "weights.npy", mmap_mode="r")
    ring = geometry["elevation_ring"]
    ring_size = int(manifest["ring_size"])
    test_rows = set(geometry["test_indices"].tolist())
    train_rings = sorted(set(ring[geometry["train_indices"]].tolist()))
    val_rings = sorted(set(ring[geometry["validation_indices"]].tolist()))
    masks = band_masks(weights.shape[1], float(manifest["sample_rate_hz"]))

    def rows(r):
        return np.arange(r * ring_size, (r + 1) * ring_size)

    def load(r):
        index = rows(r)
        if test_rows.intersection(index.tolist()):
            raise AssertionError(f"ring {r} holds reserved-test rows")
        return np.asarray(weights[index]).astype(np.complex128)

    def predictions(r):
        below, above = load(r - 1), load(r + 1)
        return {"below": below, "above": above, "average": 0.5 * (below + above)}

    def accumulate(rings):
        sums = {}
        for r in rings:
            target = np.fft.fft(load(r), axis=1)
            for name, value in predictions(r).items():
                spectrum = np.fft.fft(value, axis=1)
                entry = sums.setdefault(name, {"ping_residual": 0.0, "band": {}, "rings": {}})
                ring_entry = entry["rings"].setdefault(int(r), {"count": int(target.size)})
                ping_cross = np.sum(np.conj(spectrum) * target, axis=1)
                ping_power = np.sum(np.abs(spectrum) ** 2, axis=1)
                ping_target = np.sum(np.abs(target) ** 2, axis=1)
                entry["ping_residual"] += float(np.sum(ping_target - np.abs(ping_cross) ** 2 / np.maximum(ping_power, 1e-30)))
                for band, mask in {"full": slice(None), **masks}.items():
                    y, p = target[:, mask], spectrum[:, mask]
                    b = entry["band"].setdefault(band, {"yy": 0.0, "pp": 0.0, "py": 0j, "d1": 0.0})
                    b["yy"] += float(np.sum(np.abs(y) ** 2))
                    b["pp"] += float(np.sum(np.abs(p) ** 2))
                    b["py"] += complex(np.sum(np.conj(p) * y))
                    b["d1"] += float(np.sum(np.abs(p - y) ** 2))
                    if band in ("full", IN_BAND):
                        ring_entry[band] = {
                            "yy": float(np.sum(np.abs(y) ** 2)), "pp": float(np.sum(np.abs(p) ** 2)),
                            "py": complex(np.sum(np.conj(p) * y)), "d1": float(np.sum(np.abs(p - y) ** 2)),
                        }
        return sums

    def rel(b, g):
        return (b["yy"] - 2.0 * (np.conj(g) * b["py"]).real + abs(g) ** 2 * b["pp"]) / b["yy"]

    train_analog = [r for r in train_rings if 0 < r < ring[-1] and r - 1 in train_rings and r + 1 in train_rings]
    train_sums, val_sums = accumulate(train_analog), accumulate(val_rings)
    report = {"cache": str(root), "train_analog_rings": train_analog, "validation_rings": val_rings, "predictors": {}}
    for name in val_sums:
        tb, vb = train_sums[name]["band"], val_sums[name]["band"]
        g_train = tb["full"]["py"] / tb["full"]["pp"]
        g_val = vb["full"]["py"] / vb["full"]["pp"]
        entry = {"g_train": [g_train.real, g_train.imag], "bands": {}}
        for band, b in vb.items():
            entry["bands"][band] = {
                "energy_fraction": b["yy"] / vb["full"]["yy"],
                "single_gain_coherence": abs(b["py"]) / np.sqrt(b["pp"] * b["yy"]),
                "val_rel_mse_g1": b["d1"] / b["yy"],
                "val_rel_mse_g_train": rel(b, g_train),
                "val_rel_mse_g_val_oracle": rel(b, g_val),
                "train_analog_rel_mse_g1": tb[band]["d1"] / tb[band]["yy"],
                "train_analog_rel_mse_g_train": rel(tb[band], g_train),
            }
        entry["val_rel_mse_per_ping_gain_oracle"] = val_sums[name]["ping_residual"] / vb["full"]["yy"]
        entry["per_ring"] = {}
        for role, sums in (("validation", val_sums), ("train_analog", train_sums)):
            for r, stats in sorted(sums[name]["rings"].items()):
                entry["per_ring"].setdefault(role, {})[str(r)] = {
                    # the time-domain mean |y|^2 per ping-bin (Parseval: FFT energy / N^2 per row)
                    "mean_power": stats["full"]["yy"] / stats["count"] / weights.shape[1],
                    "full_g1": stats["full"]["d1"] / stats["full"]["yy"],
                    "full_g_train": rel(stats["full"], g_train),
                    "in_band_g1": stats[IN_BAND]["d1"] / stats[IN_BAND]["yy"],
                    "in_band_g_train": rel(stats[IN_BAND], g_train),
                }
        report["predictors"][name] = entry
    for name, entry in report["predictors"].items():
        print(f"{name}: g_train={complex(*entry['g_train']):.4f}  per-ping-gain oracle (VAL) {entry['val_rel_mse_per_ping_gain_oracle']:.4f}")
        for band, b in entry["bands"].items():
            print(
                f"  {band:>10s} kHz energy {b['energy_fraction']:.3f} single-gain coherence {b['single_gain_coherence']:.3f} "
                f"VAL g=1 {b['val_rel_mse_g1']:.4f} g_train {b['val_rel_mse_g_train']:.4f} "
                f"(g_val oracle {b['val_rel_mse_g_val_oracle']:.4f}) TRAIN-analog g=1 {b['train_analog_rel_mse_g1']:.4f}"
            )
    average = report["predictors"]["average"]["per_ring"]
    for role in ("validation", "train_analog"):
        print(f"average of r+-1, per ring ({role}): ring  mean|y|^2 per bin  full g=1  full g_train  in-band g=1  in-band g_train")
        for r, row in average[role].items():
            print(f"  {int(r):4d}  {row['mean_power']:.4f}  {row['full_g1']:.3f}  {row['full_g_train']:.3f}  "
                  f"{row['in_band_g1']:.3f}  {row['in_band_g_train']:.3f}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
