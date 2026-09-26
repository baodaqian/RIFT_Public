#!/usr/bin/env python
"""Materialize one immutable degree-6 directional-SH PublicRadar sidecar."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_cache_module():
    path = PROJECT_ROOT / "rift" / "public_radar_sh_cache.py"
    spec = importlib.util.spec_from_file_location(
        "rift_public_radar_sh_cache_materializer", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene", choices=("cvdomes_camry", "gotcha_pass2_hh"), required=True
    )
    parser.add_argument("--npz-path", required=True)
    parser.add_argument("--cache-dir", required=True)
    args = parser.parse_args()

    npz_path = Path(args.npz_path).resolve()
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    with np.load(npz_path, allow_pickle=False) as archive:
        if "viewpoint_positions" not in archive.files or "metadata_json" not in archive.files:
            raise ValueError("PublicRadar archive lacks geometry or metadata")
        positions = np.array(archive["viewpoint_positions"], copy=True)
        metadata = json.loads(str(archive["metadata_json"]))
    if metadata.get("schema") != "rift_coherent_radar_v1":
        raise ValueError("unexpected PublicRadar archive schema")
    if metadata.get("split_schema") != "rift.public_radar_interleaved_split_v2":
        raise ValueError("SH cache requires the materialized interleaved-v2 archive")
    if metadata.get("split_scene") != args.scene:
        raise ValueError("SH cache scene does not match the archive")

    cache_module = load_cache_module()
    manifest = cache_module.materialize_cache(
        args.cache_dir, positions, dataset_name=args.scene
    )
    print(json.dumps(manifest, sort_keys=True), flush=True)
    print(f"PUBLIC_RADAR_SH_CACHE_OK {Path(args.cache_dir).resolve()}", flush=True)


if __name__ == "__main__":
    main()
