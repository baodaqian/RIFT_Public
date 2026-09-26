#!/usr/bin/env python3
"""Validate the public Reed/SH-SAS dataset without unpickling any data."""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np


SH_SAS_SCENES = {"armadillo", "buddha", "bunny", "xyz_dragon"}
REED_SCENES = SH_SAS_SCENES | {"dragon", "lucy"}
AIR_SAS_FILES = {
    "system_data_arma_5k.pik",
    "system_data_arma_20k.pik",
    "system_data_bunny_5k.pik",
    "system_data_bunny_20k.pik",
}


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path("datasets/sh_sas_reed"),
    )
    parser.add_argument(
        "--test-zips",
        action="store_true",
        help="Read every member of both ZIP archives (slower).",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    airsas = root / "airsas_processed"
    simulated = root / "simulated"

    missing: list[str] = []
    extracted = airsas / "system_data_files"
    for name in sorted(AIR_SAS_FILES):
        path = extracted / name
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(str(path))

    arrays: dict[str, tuple[tuple[int, ...], str, int]] = {}
    for scene in sorted(REED_SCENES):
        path = simulated / scene / "data_full.npy"
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(str(path))
            continue
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        arrays[scene] = (array.shape, str(array.dtype), path.stat().st_size)

    for path in (
        airsas / "system_data_files.zip",
        simulated / "movies_all.zip",
        simulated / "system_data.pik",
        simulated / "gt_meshes" / "armadilo.obj",
        simulated / "gt_meshes" / "budda.obj",
        simulated / "gt_meshes" / "bunny.obj",
        simulated / "gt_meshes" / "xyz_dragon.obj",
    ):
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(str(path))

    if missing:
        fail("missing or empty files:\n  " + "\n  ".join(missing))

    shapes = {shape for shape, _dtype, _size in arrays.values()}
    dtypes = {dtype for _shape, dtype, _size in arrays.values()}
    if len(shapes) != 1:
        fail(f"transient-array shapes disagree: {shapes}")
    if len(dtypes) != 1:
        fail(f"transient-array dtypes disagree: {dtypes}")

    if args.test_zips:
        for path in (
            airsas / "system_data_files.zip",
            simulated / "movies_all.zip",
        ):
            with zipfile.ZipFile(path) as archive:
                bad_member = archive.testzip()
                if bad_member is not None:
                    fail(f"corrupt ZIP member {bad_member!r} in {path}")
                print(f"ZIP_OK {path} members={len(archive.infolist())}")

    print(f"DATASET_ROOT {root}")
    print("SH_SAS_SCENES " + " ".join(sorted(SH_SAS_SCENES)))
    for scene, (shape, dtype, size) in sorted(arrays.items()):
        membership = "SH-SAS+Reed" if scene in SH_SAS_SCENES else "Reed-only"
        print(
            f"ARRAY_OK {scene} role={membership} shape={shape} "
            f"dtype={dtype} bytes={size}"
        )
    print(f"AIRSAS_OK files={len(AIR_SAS_FILES)}")
    print(f"GT_MESH_FILES {sum(1 for _ in (simulated / 'gt_meshes').glob('*'))}")
    print("PASS public Reed/SH-SAS dataset structure is complete")


if __name__ == "__main__":
    main()
