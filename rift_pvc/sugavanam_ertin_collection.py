"""PVC twin of ``rift.sugavanam_ertin_collection``: routing only, no execution.

Contract rule 2 says the PVC frontends "emit the same argument lists ... and
only replace the script name". The safest way to guarantee that is not to copy
the argument assembly but to *delegate* to the original and substitute the one
element that names the entry point. Any future change to the SE argument list
in ``rift/sugavanam_ertin_collection.py`` is therefore picked up here
automatically and cannot silently diverge.

``add_arguments``, ``DEFAULT_RECIPE`` and ``RECIPES`` are re-exported unchanged
so the PVC RIFT-dataset frontend registers exactly the original SE flags
(``--se-stage1-only``, ``--se-recipe``).
"""
from __future__ import annotations

from pathlib import Path

from rift import sugavanam_ertin_collection as _original
from rift.sugavanam_ertin_collection import (  # re-exported unchanged
    DEFAULT_RECIPE,
    RECIPES,
    add_arguments,
)
from rift.rift_dataset import PROJECT_ROOT

__all__ = ["DEFAULT_RECIPE", "RECIPES", "add_arguments", "commands_for",
           "CUDA_ENTRY_POINT", "PVC_ENTRY_POINT"]

CUDA_ENTRY_POINT = str(PROJECT_ROOT / "train_sugavanam_ertin.py")
PVC_ENTRY_POINT = str(PROJECT_ROOT / "train_sugavanam_ertin_pvc.py")


def commands_for(name, *, dataset_root, output_root, resume=None, recipe=None,
                 config=None, device=None, check_initialization=False, manifest_path=None,
                 stage1_only=False):
    """The original SE commands with ``train_sugavanam_ertin_pvc.py`` as the entry.

    Same signature, same validation (recipe names, the explicit-``--resume``
    rule, the paper-v1-only options) and the same argument list, because it is
    the original's argument list with one element replaced.
    """
    commands = _original.commands_for(
        name, dataset_root=dataset_root, output_root=output_root, resume=resume,
        recipe=recipe, config=config, device=device,
        check_initialization=check_initialization, manifest_path=manifest_path,
        stage1_only=stage1_only)
    substituted = []
    for command in commands:
        command = list(command)
        try:
            index = command.index(CUDA_ENTRY_POINT)
        except ValueError:
            raise RuntimeError(
                "rift/sugavanam_ertin_collection.py no longer emits "
                f"{CUDA_ENTRY_POINT!r}; rift_pvc/sugavanam_ertin_collection.py must be revisited"
            ) from None
        command[index] = PVC_ENTRY_POINT
        substituted.append(command)
    return substituted
