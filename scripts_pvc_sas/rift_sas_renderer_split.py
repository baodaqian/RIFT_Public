#!/usr/bin/env python
"""Read-out-only split of the production renderer's loss into its λ gate and opacity parts (docs/RIFT_SAS_Train.md, A29).

W1's step-0 survival S5 = 0.51 (B20) mixes two factors of the production renderer
(``lambertian_ratio 0``, ``opacity_scale 500``). Each checkpoint given here is read
out (``rift_sas_checkpoint_readout.py``, TRAIN64/VAL64, never TEST) under four
renderers, with the field unchanged:

  L  linear      lambertian_ratio 1, opacity_scale 0   (the CGLS operator; T = 1, no gate)
  G  gate only   lambertian_ratio 0, opacity_scale 0
  O  opacity     lambertian_ratio 1, opacity_scale 500
  P  production  lambertian_ratio 0, opacity_scale 500 (must reproduce the checkpoint's own read-out)

and for each role the gain-free in-band quantities are compared: rel-MSE at the
TRAIN-fitted global gain and the single-gain coherence (in-band rel-MSE at the
band's own best gain is 1 - coherence^2). Survival of X against L is
(1 - X) / (1 - L) on the in-band rel-MSE at g_train_fit, as S5 was defined.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts_pvc_sas import rift_sas_checkpoint_readout as readout  # noqa: E402

IN_BAND = "12.5..27.5"
RENDERERS = (("L", 1.0, 0.0), ("G", 0.0, 0.0), ("O", 1.0, 500.0), ("P", 0.0, 500.0))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--pings", type=int, default=64)
    options = parser.parse_args(argv)
    out = Path(options.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for index, checkpoint in enumerate(options.checkpoints):
        label = f"ckpt{index}"
        entry = {"checkpoint": checkpoint, "renderers": {}}
        for name, ratio, opacity in RENDERERS:
            path = out / f"{label}_{name}.json"
            print(f"=== {label} renderer {name}: lambertian_ratio {ratio:g} opacity_scale {opacity:g} ({checkpoint}) ===", flush=True)
            readout.main([checkpoint, "--device", options.device, "--train-pings", str(options.pings),
                          "--val-pings", str(options.pings), "--lambertian-ratio", str(ratio),
                          "--opacity-scale", str(opacity), "--json", str(path)])
            report = json.loads(path.read_text())
            entry["step"] = report["step"]
            entry["renderers"][name] = {
                role: {
                    "in_band_g_train_fit": report[role][IN_BAND]["rel_mse_g_train_fit"],
                    "in_band_coherence": report[role][IN_BAND]["single_gain_coherence"],
                    "full_g_train_fit": report[role]["full"]["rel_mse_g_train_fit"],
                    "prediction_energy_in_band": report[role][IN_BAND]["prediction_energy_fraction"],
                }
                for role in ("train", "validation")
            }
        for role in ("train", "validation"):
            linear = entry["renderers"]["L"][role]["in_band_g_train_fit"]
            for name in ("G", "O", "P"):
                value = entry["renderers"][name][role]["in_band_g_train_fit"]
                entry["renderers"][name][role]["survival_vs_L"] = (1.0 - value) / (1.0 - linear) if linear < 1.0 else None
        summary.append(entry)

    for entry in summary:
        print(f"\nstep {entry['step']}: {entry['checkpoint']}")
        print(f"  {'renderer':<14}{'TRAIN in-band @g_fit':>22}{'coherence':>11}{'survival':>10}"
              f"{'VAL in-band @g_fit':>20}{'coherence':>11}{'survival':>10}")
        for name, ratio, opacity in RENDERERS:
            cells = []
            for role in ("train", "validation"):
                r = entry["renderers"][name][role]
                survival = r.get("survival_vs_L")
                cells.append(f"{r['in_band_g_train_fit']:>{22 if role == 'train' else 20}.4f}{r['in_band_coherence']:>11.3f}"
                             f"{(f'{survival:.2f}' if survival is not None else '-'):>10}")
            print(f"  {name} ({ratio:g}, {opacity:g})".ljust(16) + "".join(cells))
    (out / "renderer_split_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out / 'renderer_split_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
