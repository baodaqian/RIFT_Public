#!/usr/bin/env python
"""Read-out-only estimate of a declared output band-limit for RIFT-SAS (docs/RIFT_SAS_Train.md, A43; §4 item 2, option (a)).

Each checkpoint's raw prediction is rendered once with the unchanged
``train_sas.render_one`` on the pre-registered TRAIN subset (``--train-pings``,
default 64) and on VAL (``--val-pings``, default 0 = all 4,320 pings); reserved-test
rows are never rendered. The prediction is then scored in variants with whole
FFT bands of the *output* set to zero (the target is never changed):

  none      the checkpoint as it is
  neg       zero < 0 kHz (an analytic output)
  neg_high  zero < 0 kHz and 27.5-50 kHz
  in_band   keep 12.5-27.5 kHz only
  pb_neg_high  zero < 0 and 27.5-50 kHz; separate gains for 0-12.5 and 12.5-27.5 kHz (§4 option (b), A46)
  pb_all       no band zeroed; a separate gain for each of the four bands

For each variant, full-band rel-MSE is reported
  * at the checkpoint's stored g (information: that g was fitted for the unlimited output), and
  * at complex gains refitted on the TRAIN subset for the band-limited output (``g_train_fit``;
    the comparable score, as §2's gain convention): one global gain over the kept bands, or
    for the ``pb_`` variants one gain per band group,
with the per-band contributions (target-energy fraction x band rel-MSE, summing to the
full score). Variant ``none`` at stored g on all VAL pings must reproduce the
trainer's authoritative full-VAL score of the same checkpoint. This is a static
estimate: the model is not retrained with the band-limit, so its gradients never
saw it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train_sas  # noqa: E402
from rift.sas_dataset import load_sas_cache  # noqa: E402
from scripts_pvc_sas.rift_sas_checkpoint_readout import render_spectra  # noqa: E402
from scripts_pvc_sas.rift_sas_references import band_masks  # noqa: E402

BANDS = ("-50..0", "0..12.5", "12.5..27.5", "27.5..50")
# name: (zeroed output bands, gain groups; None = one global gain over the kept bands)
VARIANTS = {
    "none": ((), None),
    "neg": (("-50..0",), None),
    "neg_high": (("-50..0", "27.5..50"), None),
    "in_band": (("-50..0", "0..12.5", "27.5..50"), None),
    "pb_neg_high": (("-50..0", "27.5..50"), (("0..12.5",), ("12.5..27.5",))),
    "pb_all": ((), tuple((band,) for band in BANDS)),
}


def fit_gains(target, prediction, keep, groups, masks):
    """Per-bin gain vector (zero outside ``keep``) with one least-squares gain per group, fitted on these rows."""
    gains = np.zeros(target.shape[1], dtype=np.complex128)
    fitted = {}
    for group in groups or (None,):
        select = keep if group is None else keep & np.any([masks[band] for band in group], axis=0)
        p, y = prediction[:, select], target[:, select]
        g = complex(np.sum(np.conj(p) * y) / np.sum(np.abs(p) ** 2))
        gains[select] = g
        fitted["kept" if group is None else "+".join(group)] = [g.real, g.imag]
    return gains, fitted


def score(target, prediction, gain, masks):
    """Full rel-MSE and per-band contributions of ``gain * prediction``; gain is a scalar or a per-bin vector."""
    total = float(np.sum(np.abs(target) ** 2))
    error = np.abs(gain * prediction - target) ** 2
    out = {"full": float(np.sum(error)) / total}
    for band, mask in masks.items():
        out[band] = float(np.sum(error[:, mask])) / total   # contribution; sums to full
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--train-pings", type=int, default=64)
    parser.add_argument("--val-pings", type=int, default=0, help="evenly spaced VAL pings; 0 = all")
    parser.add_argument("--json", required=True)
    options = parser.parse_args(argv)
    device = torch.device(options.device)
    report = []
    for checkpoint in options.checkpoints:
        started = time.time()
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        args = argparse.Namespace(**state["args"])
        args.device = str(device)
        cache = load_sas_cache(args.cache)
        model = train_sas.build_model(args, cache, device)
        calibration = train_sas.build_calibration(state["calibration_mode"], device)
        model.load_state_dict(state["model_state_dict"])
        calibration.load_state_dict(state["calibration_state_dict"])
        model.eval()
        calibration.eval()
        with torch.no_grad():
            stored = complex(calibration(torch.ones((), dtype=torch.complex64, device=device)).cpu().item())
        masks = band_masks(cache.num_bins, float(cache.manifest["sample_rate_hz"]))
        covered = np.zeros(cache.num_bins, dtype=int)
        for mask in masks.values():
            covered += mask
        if not np.all(covered == 1):
            raise AssertionError("the bands must partition the FFT bins")
        train_pings = train_sas.select_eval_indices(cache.train_indices, options.train_pings)
        val_pings = train_sas.select_eval_indices(cache.validation_indices, options.val_pings)
        test_rows = set(cache.test_indices.tolist())
        if test_rows.intersection(train_pings.tolist()) or test_rows.intersection(val_pings.tolist()):
            raise AssertionError("a reserved-test row was selected")
        y_train, p_train = render_spectra(model, calibration, cache, train_pings, args, device)
        y_val, p_val = render_spectra(model, calibration, cache, val_pings, args, device)
        entry = {"checkpoint": checkpoint, "step": int(state["step"]), "stored_g": [stored.real, stored.imag],
                 "train_pings": int(train_pings.size), "val_pings": int(val_pings.size), "variants": {}}
        for name, (zeroed, groups) in VARIANTS.items():
            keep = np.ones(cache.num_bins, dtype=bool)
            for band in zeroed:
                keep &= ~masks[band]
            gains, fitted = fit_gains(y_train, p_train, keep, groups, masks)
            stored_gains = stored * keep
            entry["variants"][name] = {
                "zeroed_bands": list(zeroed),
                "g_train_fit": fitted,
                "train": {"stored_g": score(y_train, p_train, stored_gains, masks), "g_train_fit": score(y_train, p_train, gains, masks)},
                "validation": {"stored_g": score(y_val, p_val, stored_gains, masks), "g_train_fit": score(y_val, p_val, gains, masks)},
            }
        entry["elapsed_seconds"] = time.time() - started
        report.append(entry)

        print(f"\nstep {entry['step']}: {checkpoint}  (TRAIN {train_pings.size} pings for the gain; VAL {val_pings.size} pings; "
              f"stored g {stored:.4g}; {entry['elapsed_seconds']:.0f}s)")
        print(f"  {'variant':<12}{'|g_fit|':>9}{'VAL full @stored':>18}{'VAL full @g_fit':>17}{'TRAIN full @g_fit':>19}"
              "  | VAL contributions @g_fit: " + " ".join(f"{b:>10}" for b in masks))
        for name, v in entry["variants"].items():
            vf = v["validation"]["g_train_fit"]
            in_band_gain = next(g for key, g in v["g_train_fit"].items() if key in ("kept", "12.5..27.5"))
            print(f"  {name:<12}{abs(complex(*in_band_gain)):>9.4f}{v['validation']['stored_g']['full']:>18.4f}{vf['full']:>17.4f}"
                  f"{v['train']['g_train_fit']['full']:>19.4f}  | " + " " * 26 + " ".join(f"{vf[b]:>10.4f}" for b in masks))
        del model, calibration, y_val, p_val
    Path(options.json).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {options.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
