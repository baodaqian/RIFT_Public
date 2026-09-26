#!/usr/bin/env python
"""TRAIN/VAL read-out of a sonar checkpoint with gain conventions and band split.

Renders pre-registered, evenly spaced TRAIN and VAL subsets
(``train_sas.select_eval_indices``) over all cached bins with the unchanged
``train_sas.render_one`` and reports complex rel-MSE:
  * at the checkpoint's stored global gain (what ``train_sas.evaluate`` scores);
  * at the best global gain fitted on the TRAIN subset (full band), applied
    unchanged to VAL (the stored gain lags under Adam);
  * per-ping gains: report-only.
Each quantity is also split by band (FFT over the bins) at the same gains, with
the energy fractions of target and prediction and their coherence.

C5 of docs/RIFT_SAS_Train.md: the training ping draws are replayed from the
seed (numpy ``default_rng``; ping choice then ``select_bins``) up to the
checkpoint step, the replayed generator state is compared with the saved one,
and TRAIN is reported separately for seen and unseen pings. Because the pooled
score is energy-weighted and the mean power falls about 40x from the lowest to
the highest rings, each role is also split into elevation groups (rings 0-39,
40-79, 80-119) at the same gains: a diagnostic beside the pooled score, which
stays the score. Reserved-test rows are never rendered.

Renderer override (A29, the S5 split of B20/A25): ``--lambertian-ratio`` and
``--opacity-scale`` replace the checkpoint's values for rendering only, so one
field can be read under the production renderer and under the λ-gate-free or
opacity-free renderers. The report records the renderer used. Under an override
the stored g was fitted for another renderer and is information only; the
gain-free quantities (g_train_fit, single-gain coherence) are the comparison.
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
from scripts_pvc_sas.rift_sas_references import band_masks  # noqa: E402


ELEVATION_GROUPS = ((0, 39), (40, 79), (80, 119))


def replay_seen(state, args, cache):
    rng = np.random.default_rng(int(args.seed))
    dummy = np.empty(cache.num_bins)
    seen = set()
    for _step in range(int(state["step"])):
        for _ in range(int(getattr(args, "pings_per_step", 1))):
            seen.add(int(rng.choice(cache.train_indices)))
            train_sas.select_bins(rng, dummy, int(args.max_bins))
    return seen, rng.bit_generator.state == state["rng_state"]


def render_spectra(model, calibration, cache, pings, args, device):
    bins = np.arange(cache.num_bins)
    targets, predictions = [], []
    with torch.no_grad():
        for ping in pings:
            _loss, _metrics, aux = train_sas.render_one(model, calibration, cache, int(ping), bins, args, device)
            predictions.append(np.fft.fft(aux["calibration_raw"].to(torch.complex128).cpu().numpy()))
            targets.append(np.fft.fft(aux["calibration_target"].to(torch.complex128).cpu().numpy()))
    return np.asarray(targets), np.asarray(predictions)


def summarize(target, prediction, gains, masks):
    out = {"pings": int(target.shape[0])}
    ping_cross = np.sum(np.conj(prediction) * target, axis=1)
    ping_power = np.sum(np.abs(prediction) ** 2, axis=1)
    ping_target = np.sum(np.abs(target) ** 2, axis=1)
    out["rel_mse_per_ping_gain_oracle"] = float(
        np.sum(ping_target - np.abs(ping_cross) ** 2 / np.maximum(ping_power, 1e-30)) / np.sum(ping_target)
    )
    total_target = float(np.sum(np.abs(target) ** 2))
    total_prediction = float(np.sum(np.abs(prediction) ** 2))
    for band, mask in {"full": slice(None), **masks}.items():
        y, p = target[:, mask], prediction[:, mask]
        yy, pp = float(np.sum(np.abs(y) ** 2)), float(np.sum(np.abs(p) ** 2))
        py = complex(np.sum(np.conj(p) * y))
        entry = {
            "target_energy_fraction": yy / total_target,
            "prediction_energy_fraction": pp / max(total_prediction, 1e-30),
            "single_gain_coherence": abs(py) / max(np.sqrt(pp * yy), 1e-30),
        }
        for name, g in gains.items():
            entry[f"rel_mse_{name}"] = (yy - 2.0 * (np.conj(g) * py).real + abs(g) ** 2 * pp) / yy
        out[band] = entry
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-pings", type=int, default=64, help="evenly spaced TRAIN pings; 0 = all")
    parser.add_argument("--val-pings", type=int, default=64, help="evenly spaced VAL pings; 0 = all")
    parser.add_argument("--json", default=None)
    parser.add_argument("--lambertian-ratio", type=float, default=None, help="render with this ratio instead of the checkpoint's")
    parser.add_argument("--opacity-scale", type=float, default=None, help="render with this opacity instead of the checkpoint's")
    options = parser.parse_args(argv)
    device = torch.device(options.device)
    state = torch.load(options.checkpoint, map_location="cpu", weights_only=False)
    args = argparse.Namespace(**state["args"])
    args.device = str(device)
    renderer = {"checkpoint_lambertian_ratio": float(args.lambertian_ratio),
                "checkpoint_opacity_scale": float(args.opacity_scale)}
    if options.lambertian_ratio is not None:
        args.lambertian_ratio = options.lambertian_ratio
    if options.opacity_scale is not None:
        args.opacity_scale = options.opacity_scale
    renderer.update(lambertian_ratio=float(args.lambertian_ratio), opacity_scale=float(args.opacity_scale))
    renderer["overridden"] = (renderer["lambertian_ratio"] != renderer["checkpoint_lambertian_ratio"]
                              or renderer["opacity_scale"] != renderer["checkpoint_opacity_scale"])
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
    seen, replay_ok = replay_seen(state, args, cache)
    test_rows = set(cache.test_indices.tolist())

    started = time.time()
    train_pings = train_sas.select_eval_indices(cache.train_indices, options.train_pings)
    val_pings = train_sas.select_eval_indices(cache.validation_indices, options.val_pings)
    if test_rows.intersection(train_pings.tolist()) or test_rows.intersection(val_pings.tolist()):
        raise AssertionError("a reserved-test row was selected")
    y_train, p_train = render_spectra(model, calibration, cache, train_pings, args, device)
    fitted = complex(np.sum(np.conj(p_train) * y_train) / np.sum(np.abs(p_train) ** 2))
    gains = {"stored_g": stored, "g_train_fit": fitted}
    report = {
        "checkpoint": str(options.checkpoint),
        "step": int(state["step"]),
        "model": args.model,
        "renderer": renderer,
        "stored_g": [stored.real, stored.imag],
        "g_train_fit": [fitted.real, fitted.imag],
        "rng_replay_matches_checkpoint": bool(replay_ok),
        "train_pings_seen_by_step": len(seen),
        "train_fraction_seen": len(seen) / cache.train_indices.size,
        "train": summarize(y_train, p_train, gains, masks),
    }
    seen_mask = np.asarray([int(p) in seen for p in train_pings])
    report["train_subset_seen_count"] = int(seen_mask.sum())
    for label, mask in (("train_seen", seen_mask), ("train_unseen", ~seen_mask)):
        if mask.any():
            report[label] = summarize(y_train[mask], p_train[mask], gains, masks)
    ring_size = int(cache.manifest["ring_size"])
    y_val = p_val = None
    if val_pings.size:
        y_val, p_val = render_spectra(model, calibration, cache, val_pings, args, device)
        report["validation"] = summarize(y_val, p_val, gains, masks)
    for role, pings, y, p in (("train", train_pings, y_train, p_train), ("validation", val_pings, y_val, p_val)):
        if y is None:
            continue
        rings = np.asarray(pings) // ring_size
        for lo, hi in ELEVATION_GROUPS:
            mask = (rings >= lo) & (rings <= hi)
            if mask.any():
                report.setdefault("elevation_groups", {}).setdefault(role, {})[f"rings_{lo}_{hi}"] = summarize(
                    y[mask], p[mask], gains, masks
                )
    report["elapsed_seconds"] = time.time() - started

    if renderer["overridden"]:
        print(f"renderer OVERRIDE: lambertian_ratio {renderer['lambertian_ratio']:g} opacity_scale {renderer['opacity_scale']:g} "
              f"(checkpoint {renderer['checkpoint_lambertian_ratio']:g} / {renderer['checkpoint_opacity_scale']:g}); stored g is information only")
    print(f"step {report['step']} stored g {stored:.4g}  g_train_fit {fitted:.4g}  replay matches: {replay_ok}  "
          f"TRAIN seen {len(seen)}/{cache.train_indices.size}; subset seen {int(seen_mask.sum())}/{train_pings.size}")
    for role in ("train", "train_seen", "train_unseen", "validation"):
        if role not in report:
            continue
        entry = report[role]
        print(f"{role} ({entry['pings']} pings; per-ping-gain oracle {entry['rel_mse_per_ping_gain_oracle']:.4f})")
        for band in ("full", *masks):
            b = entry[band]
            print(f"  {band:>10s} kHz target {b['target_energy_fraction']:.3f} pred {b['prediction_energy_fraction']:.3f} "
                  f"single-gain coherence {b['single_gain_coherence']:.3f} rel@stored {b['rel_mse_stored_g']:.4f} rel@g_train {b['rel_mse_g_train_fit']:.4f}")
    for role, groups in report.get("elevation_groups", {}).items():
        for group, entry in groups.items():
            full, in_band = entry["full"], entry["12.5..27.5"]
            print(f"{role} {group} ({entry['pings']} pings): full rel@g_train {full['rel_mse_g_train_fit']:.4f} "
                  f"in-band rel@g_train {in_band['rel_mse_g_train_fit']:.4f} (stored g {full['rel_mse_stored_g']:.4f})")
    if options.json:
        Path(options.json).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
