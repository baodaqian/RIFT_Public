"""Native GOTCHA RadarSplat backend on the PVC torch-mirror renderer.

``rift/radarsplat_gotcha.py`` names CUDA in exactly one place: its
``load_cuda_reference`` import (used once, in ``run_gotcha``). Importing this
module rebinds that name in the unchanged module to
``rift_pvc.radarsplat_xpu_backend.load_xpu_reference``; the engine it calls
(``rift.radarsplat_release_training``) already carries the PVC twins after
``rift_pvc.radarsplat_release_training`` is imported. ``sector_power`` is
generic torch (complex128 on the requested device). ``run_gotcha`` keeps its
signature so ``train_gotcha_dataset_pvc.py`` binds it like the CUDA frontend.
"""
from __future__ import annotations

from pathlib import Path

import rift.radarsplat_gotcha as _backend
import rift_pvc.radarsplat_release_training as _engine_pvc  # noqa: F401  (installs the engine twins)
from rift.radarsplat_gotcha import (  # noqa: F401  (re-exports)
    CONTROL_FILE, CONTROL_SCHEMA, GOTCHAPowerCache, adapter_config, cache_from_run, planning, sampling,
    sector_power,
)
from rift_pvc.radarsplat_xpu_backend import (
    BACKEND_IDENTITY, check_resume_sidecar, load_xpu_reference, read_sidecar, write_sidecar,
)


def install() -> None:
    _backend.load_cuda_reference = load_xpu_reference
    _engine_pvc.install()


def run_gotcha(*, dataset, output_dir, config, device, resume):
    install()
    output = Path(output_dir)
    # The original orchestrator skips engine.train for completed heads. Check
    # every backend boundary here, before it can read/prepare another head.
    check_resume_sidecar(output)
    for polarization in dataset.polarizations:
        check_resume_sidecar(output / polarization / "checkpoints")
    result = _backend.run_gotcha(dataset=dataset, output_dir=output, config=config, device=device, resume=resume)
    # Do this after the original fresh-output/identity checks. Conversion-only
    # recovery remains compatible; it has no learned renderer state to mix.
    write_sidecar(output)
    result["pvc"] = dict(read_sidecar(output))
    return result


install()

__all__ = ["CONTROL_FILE", "CONTROL_SCHEMA", "GOTCHAPowerCache", "adapter_config", "cache_from_run", "planning",
           "sampling", "sector_power", "run_gotcha", "install", "BACKEND_IDENTITY"]
