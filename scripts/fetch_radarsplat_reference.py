#!/usr/bin/env python3
"""Fetch official RadarSplat source and check its inventory; no hash checks."""
from __future__ import annotations
import argparse
import io
from pathlib import Path
import sys
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rift.radarsplat_release import COMMIT, REFERENCE_ROOT, reference_contract, verify_reference


def materialize(payload, root, inventory):
    root = Path(root).resolve()
    # Check the file inventory before writing. Never overwrite an existing
    # checkout, follow archive links or install/build a dependency implicitly.
    files = {}
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for member in archive.getmembers():
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError("Archive links/special files are not permitted")
            relative = Path(*Path(member.name).parts[1:])
            if relative.is_absolute() or ".." in relative.parts or str(relative) in files:
                raise ValueError("Invalid archive path")
            data = archive.extractfile(member).read()
            if str(relative) not in inventory:
                raise ValueError(f"Unexpected source file: {relative}")
            target = root/relative
            if any(p.is_symlink() for p in (target, *target.parents)):
                raise ValueError("Reference destination must not traverse symlinks")
            files[str(relative)] = data
    if set(files) != set(inventory):
        raise ValueError("Incomplete reference archive")
    for relative, data in files.items():
        target = root/relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use a previously downloaded archive")
    parser.add_argument("--root", type=Path, default=REFERENCE_ROOT)
    parser.add_argument("--glm-archive", type=Path, help="Use the pinned GLM archive without network")
    args = parser.parse_args()
    payload = (args.archive.read_bytes() if args.archive else urllib.request.urlopen(
        f"https://codeload.github.com/umautobots/radarsplat/tar.gz/{COMMIT}", timeout=120).read())
    contract = reference_contract()
    materialize(payload, args.root, contract["files_sha256"])
    glm_path = "gsplat/cuda/csrc/third_party/glm"
    commit = contract["submodules"][glm_path]
    glm_payload = (args.glm_archive.read_bytes() if args.glm_archive else urllib.request.urlopen(
        f"https://codeload.github.com/g-truc/glm/tar.gz/{commit}", timeout=120).read())
    materialize(glm_payload, args.root/glm_path, contract["glm_files_sha256"])
    print(verify_reference(args.root, cuda_dependencies=True))


if __name__ == "__main__":
    main()
