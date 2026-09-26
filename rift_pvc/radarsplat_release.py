"""PVC twin of ``rift/radarsplat_release.py``.

The unchanged module supplies the initializer, preprocessing, renderer adapter
and loss. Only the two accelerator-specific names differ here:

* ``create_scene`` defaults ``device`` to ``rift_pvc.accelerator.device()``
  (the original defaults to ``"cuda"``);
* ``load_cuda_reference`` is ``rift_pvc.radarsplat_xpu_backend.load_xpu_reference``
  (the original requires CUDA and imports the compiled fused-SSIM).

Everything else is re-exported from the original so callers written against
``rift.radarsplat_release`` can import this module instead.
"""
from __future__ import annotations

import numpy as np
import torch

from rift.radarsplat_release import (  # noqa: F401  (re-exports)
    COMMIT, MANIFEST, PROFILES, REFERENCE_ROOT, SCHEMA, ReleasedRenderer,
    model_recipe, position_scheduler, profile_from_identity, recipe, reference_contract, release_loss,
    source_functions, verify_reference,
)
from rift.radarsplat_release import ReleasedPreprocessing as _ReleasedPreprocessing
from rift.radarsplat_release import create_scene as _create_scene
from rift_pvc import accelerator
from rift_pvc.radarsplat_xpu_backend import (  # noqa: F401
    BACKEND_IDENTITY, SSIM_IDENTITY, load_xpu_reference, preprocessing_functions,
)


class ReleasedPreprocessing(_ReleasedPreprocessing):
    """The original training-only denoising/periodic fit; its FFT runs on the accelerator (F-dev1).

    The parent constructor loads the release's ``FFT``/``multipath_modeling``
    ASTs, whose ``FFT`` moves the image to CUDA; they are replaced here by the
    same functions with that one literal pointing at ``device``.
    """

    def __init__(self, cache, *, root=REFERENCE_ROOT, device=None, **kwargs):
        super().__init__(cache, root=root, **kwargs)
        self.device = accelerator.device() if device is None else torch.device(device)
        self.functions = preprocessing_functions(self.device, root=root)


def create_scene(*, scene_scale, scene_center, device=None, seed=42, root=REFERENCE_ROOT, num_points=20_000):
    """The original initializer; the scene is drawn on the CPU generator and moved afterwards."""
    device = accelerator.device() if device is None else device
    return _create_scene(scene_scale=scene_scale, scene_center=np.asarray(scene_center, dtype=np.float64),
                         device=device, seed=seed, root=root, num_points=num_points)


load_cuda_reference = load_xpu_reference  # the name callers of the CUDA module use
load_reference = load_xpu_reference

__all__ = [
    "COMMIT", "MANIFEST", "PROFILES", "REFERENCE_ROOT", "SCHEMA", "ReleasedPreprocessing", "ReleasedRenderer",
    "model_recipe", "position_scheduler", "profile_from_identity", "recipe", "reference_contract", "release_loss",
    "source_functions", "verify_reference", "create_scene", "load_cuda_reference", "load_reference",
    "load_xpu_reference", "BACKEND_IDENTITY", "SSIM_IDENTITY",
]
