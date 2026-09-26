#!/usr/bin/env python
"""PVC (Intel XPU) entry point for the Sugavanam--Ertin B787-3200 Stage 1.

Audit C.4: ``train_sugavanam_ertin_stage1.py`` has **zero** CUDA touches. It is
a thin preflight wrapper that validates the sealed B787-3200 identity and then
hands all compute to ``train.main(frozen_train_argv(...))``.

So the only adaptation is the accelerator itself: install the RIFT agent's
``train_pvc`` rebinding (``init_distributed`` -> xccl, ``set_seed``,
``capture_rng_state``/``restore_rng_state`` with the XPU payload key) into the
``train`` module *before* the wrapper calls it. The wrapper's frozen argv,
its recipe, its resume contract and its Stage-1 bundle validation are the
originals, untouched and not copied.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_pvc  # noqa: E402  (accelerator rebinding for the shared trainer)
import train_sugavanam_ertin_stage1 as _original  # noqa: E402  (unchanged wrapper)

# Re-exported so callers and tests see the original contract constants.
CHECKPOINT_NAME = _original.CHECKPOINT_NAME
CHECKPOINT_ROOT = _original.CHECKPOINT_ROOT
GENERIC_OUTPUT_DIR = _original.GENERIC_OUTPUT_DIR
GENERIC_FINAL_PATH = _original.GENERIC_FINAL_PATH
GENERIC_LATEST_PATH = _original.GENERIC_LATEST_PATH
build_parser = _original.build_parser
parse_args = _original.parse_args


def main(argv=None):
    train_pvc.check_backend()
    train_pvc.install()
    from rift_pvc import accelerator
    print(f"train_sugavanam_ertin_stage1_pvc.py: {accelerator.describe()}", flush=True)
    return _original.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
