#!/usr/bin/env python3
"""Print Reed's train-only raw-trace scale for the first AirSAS subset.

The scale is computed after the official crop, before any Hilbert conversion,
using only the declared training rings.  The value is passed into Reed's
deconvolution INR, where it is applied before GPU transfer and optimization.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prepare_airsas_cache import NUM_RINGS, RING_SIZE, _load_system_data, _schema_dict, _split_indices


NUM_TRANSMISSIONS = NUM_RINGS * RING_SIZE


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system-data", type=Path, required=True)
    parser.add_argument("--reed-root", type=Path, default=Path("external/Reed_SAS_reference"))
    parser.add_argument("--max-transmissions", type=int, default=NUM_TRANSMISSIONS)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.max_transmissions != NUM_TRANSMISSIONS:
        raise ValueError("the train-only AirSAS frontend is fixed to the first 43200 transmissions")
    system = _load_system_data(args.system_data, args.reed_root)
    raw = np.asarray(system["wfm_data"])
    crop = _schema_dict(system["crop_settings"])
    num_samples = int(crop["num_samples"])
    if raw.ndim != 2 or raw.shape[0] < NUM_TRANSMISSIONS:
        raise RuntimeError("wfm_data must contain at least the first 43200 transmissions")
    if raw.shape[1] != num_samples:
        raw = raw[:, int(crop["min_sample"]):int(crop["min_sample"]) + num_samples]
    if raw.shape != (raw.shape[0], num_samples) or not np.isfinite(raw[:NUM_TRANSMISSIONS]).all():
        raise RuntimeError("cropped raw AirSAS traces are not finite with the expected shape")
    train, _validation, _test, _train_rings, _validation_rings, _test_rings = _split_indices(
        NUM_TRANSMISSIONS, RING_SIZE, NUM_RINGS
    )
    scale = float(np.max(np.abs(raw[train])))
    if not np.isfinite(scale) or scale <= 0.0:
        raise RuntimeError("cropped raw TRAIN traces have no finite positive amplitude")
    print(f"{scale:.17g}")


if __name__ == "__main__":
    main()
