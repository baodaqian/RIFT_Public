#!/usr/bin/env python
"""Static contract checks for the fresh Stage-2 CPU environment launcher v2.

This validator intentionally has no Torch, NumPy, SciPy, scikit-image, B787,
checkpoint, or geometry dependency.  It confirms only that the new launcher
keeps the RIFT environment first and adds the established external
scikit-image site in a scoped, inspectable way.  The launcher itself performs
the allocated import and marching-cubes preflight.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "slurm" / "validate_sugavanam_ertin_b7873200_stage2_v2.sbatch"


def require(source: str, fragment: str) -> None:
    if fragment not in source:
        raise AssertionError(f"missing required launcher/package fragment: {fragment!r}")


def forbid(source: str, fragment: str) -> None:
    if fragment in source:
        raise AssertionError(f"forbidden environment mutation fragment present: {fragment!r}")


def main() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")

    if not launcher.startswith("#!/bin/bash\n"):
        raise AssertionError("v2 launcher must retain an executable Bash shebang")
    if "\r" in launcher:
        raise AssertionError("v2 launcher must use LF line endings")

    required_launcher = (
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --export=NONE",
        "SE_B7873200_STAGE2_ENVIRONMENT_V2_PASS",
        "SE_B7873200_ALLOCATED_CPU_CONTRACT_V2_PASS",
        "conda activate RIFT",
        'RIFT_SITE="$CONDA_PREFIX/lib/python3.10/site-packages"',
        "EXPORT_SITE=/storage/home/hcoda1/1/dbao31/r-jromberg3-0/daqian_software/conda_envs/sensor_fusion/lib/python3.10/site-packages",
        'export PYTHONPATH="$PROJECT_ROOT:$RIFT_SITE:$EXPORT_SITE"',
        "from skimage.measure import marching_cubes",
        "RIFT site-packages must precede the scoped scikit-image site",
        "require_origin(numpy, rift_site, \"numpy\")",
        "require_origin(scipy, rift_site, \"scipy\")",
        "require_origin(torch, rift_site, \"torch\")",
        "require_origin(skimage, export_site, \"skimage\")",
        "run_validator environment_static",
        "scripts/validate_sugavanam_ertin_b7873200_stage1.py",
        "scripts/validate_sugavanam_ertin_b7873200_stage1_operator.py",
        "scripts/validate_sugavanam_ertin_b7873200_stage2_static.py",
        "scripts/validate_sugavanam_ertin_b7873200_stage2_v1.py",
    )
    for fragment in required_launcher:
        require(launcher, fragment)

    for forbidden_fragment in (
        "pip install",
        "conda install",
        "conda env create",
        "conda env update",
        "mamba install",
        "#SBATCH --gres",
        "--npz-path",
        ".npz",
        "--device cuda",
        "train_sugavanam_ertin_b7873200_stage2_v2.py",
    ):
        forbid(launcher, forbidden_fragment)

    if launcher.count("<<'PY'") != 1 or "\nPY\n\nvalidation_parent=" not in launcher:
        raise AssertionError("v2 launcher must retain one bounded import-preflight heredoc")

    print(
        "SE_B7873200_STAGE2_ENVIRONMENT_STATIC_V2_PASS: "
        "scoped-launcher checks",
        flush=True,
    )


if __name__ == "__main__":
    main()
