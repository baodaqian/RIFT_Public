#!/usr/bin/env python3
"""Validate a complete AirSAS5k cache before either comparison arm runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rift.airsas_contract import validate_cache_5k


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache")
    args = parser.parse_args()
    manifest = validate_cache_5k(args.cache)
    print(
        "PASS AirSAS5k cache identity "
        f"{manifest['dataset_identity']} pings={manifest['num_pings']} "
        f"original_bins={manifest['original_num_bins']}"
    )


if __name__ == "__main__":
    main()
