#!/usr/bin/env python
"""PVC (Intel XPU) entry point for the Sugavanam--Ertin baseline.

Same CLI as ``train_sugavanam_ertin.py``. ``--recipe paper-v1`` -- the
production lane, and the only one the RIFT and GOTCHA frontends use -- is
routed to ``rift_pvc.sugavanam_ertin_paper_workflow``, which is the unchanged
scientific workflow with its ``device`` default resolved to the active
accelerator. See RIFT_PVC_Adaptation.md section C.3 for the audit.

``--recipe legacy-full`` is **refused here**. It is the sealed B787-3200
"Inferno CUDA lane": ``train_sugavanam_ertin.py`` L602 itself states that
identity, its resume contract is bound to canonical CUDA checkpoints, and its
``_enforce_resource`` gate (L724-727) requires finite ``cuda_peak_*``
measurements. Running it on XPU would mint a new identity under a sealed name,
so this entry point declines and names the CUDA lane instead. Nothing about
that lane is modified.

``run_gotcha`` is exported with the signature ``train_gotcha_dataset_pvc.py``
imports, exactly as ``train_gotcha_dataset.py`` imports it from
``train_sugavanam_ertin.py``.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rift_pvc import accelerator  # noqa: E402
from rift_pvc import sugavanam_ertin_paper_workflow as paper_workflow_pvc  # noqa: E402
import train_sugavanam_ertin as _original  # noqa: E402  (unchanged; imported for its contracts)

LEGACY_REFUSAL = (
    "--recipe legacy-full is the sealed B787-3200 CUDA identity (see "
    "train_sugavanam_ertin.py L602: 'the full package is an Inferno CUDA lane'). "
    "Its resume contract and resource gate are bound to CUDA checkpoints and "
    "cuda_peak_* measurements, so the PVC entry point does not reproduce it under "
    "the same name. Run it with train_sugavanam_ertin.py in the CUDA environment; "
    "on PVC use --recipe paper-v1."
)


def run_gotcha(*, dataset, output_dir, config, device, resume):
    """GOTCHA run function, imported by ``train_gotcha_dataset_pvc.py``.

    Mirrors ``train_sugavanam_ertin.run_gotcha`` (L97-101): a thin delegation to
    the paper workflow, here its PVC twin.
    """
    return paper_workflow_pvc.run_gotcha(dataset=dataset, output_dir=output_dir,
                                         config=config, device=device, resume=resume)


# Must stay a *literal* mapping: train_gotcha_dataset_pvc.py::backend_registry
# reads this declaration with ast.literal_eval, without importing the module.
# It is the original's declaration verbatim; test_run_gotcha_signature_is_identical
# asserts equality with train_sugavanam_ertin.GOTCHA_BACKEND so it cannot drift.
GOTCHA_BACKEND = {
    "schema": "rift_gotcha_backend_v1",
    "method": "sugavanam_ertin",
    "callable": "run_gotcha",
    "selection_unit": "pass_sector",
    "joint_passes": True,
    "native_frequency_policy": "ragged_exact",
    "polarizations": ["hh"],
    "metric_domain": "stage1_native_complex_diagnostic_stage2_SDF_geometry",
    "fidelity_status": "published_initialization_unresolved_not_benchmark_ready",
}


def check_backend():
    """Refuse to start outside the PVC lane unless explicitly overridden."""
    allowed = {"xpu"} | {b.strip() for b in os.environ.get("RIFT_PVC_ALLOW_BACKEND", "").split(",") if b.strip()}
    backend = accelerator.backend()
    if backend not in allowed:
        raise RuntimeError(
            f"train_sugavanam_ertin_pvc.py is the PVC entry point and found backend {backend!r}; "
            "use train_sugavanam_ertin.py on CUDA, or set RIFT_PVC_ALLOW_BACKEND=cpu|cuda to override")
    return backend


def main(argv=None):
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--recipe", choices=["legacy-full", "paper-v1"], default="legacy-full")
    selector.add_argument("--dry-run", action="store_true")
    selector.add_argument("--check-initialization", action="store_true")
    selected, _ = selector.parse_known_args(argv)
    if selected.recipe != "paper-v1":
        raise SystemExit(LEGACY_REFUSAL)
    if selected.dry_run or selected.check_initialization:
        # Read-only planner and the bounded CPU initialization probe. Both run
        # without any accelerator, so they stay usable on a login node -- the
        # probe is the intended tool for reporting the initialization gate.
        return paper_workflow_pvc.main(argv)
    backend = check_backend()
    print(f"train_sugavanam_ertin_pvc.py: {accelerator.describe()}", flush=True)
    if backend == "xpu":
        print("train_sugavanam_ertin_pvc.py: PYTORCH_DEBUG_XPU_FALLBACK="
              f"{os.environ.get('PYTORCH_DEBUG_XPU_FALLBACK', 'unset')} SYCL_CACHE_PERSISTENT="
              f"{os.environ.get('SYCL_CACHE_PERSISTENT', 'unset')}", flush=True)
    # The workflow parser owns --recipe too, so the full argv is passed through
    # unchanged, exactly as train_sugavanam_ertin.py L1749 does.
    return paper_workflow_pvc.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
