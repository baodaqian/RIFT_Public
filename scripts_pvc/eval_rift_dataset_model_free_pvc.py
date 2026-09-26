#!/usr/bin/env python3
"""PVC twin of ``scripts/eval_rift_dataset_model_free.py`` (the ``fsh`` / ``mfbp`` model-free baselines).

``train_rift_dataset_pvc.py`` maps the planned CUDA command to this entry point. The unchanged script is
imported and run as is: the sealed-role reader, the finite-SH sweep and the train-only coherent
matched-filter backprojection (``mfbp``: phase-only matched filter, ``phase_sign=-1``, ``range_nufft``,
float64, on the 48^3 cell-centred evaluator lattice over +-0.15 m) are the original's. The only
accelerator-specific name is the ``--device`` default: the original defaults to ``cuda``; here
``--device <accelerator.device()>`` is injected when the caller omits the flag (``cpu`` under
``RIFT_ACCELERATOR=cpu``, as on the CPU partition).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rift_pvc import accelerator  # noqa: E402
import scripts.eval_rift_dataset_model_free as _model_free  # noqa: E402  (unchanged script)


def _device_in(argv) -> bool:
    return any(a == "--device" or a.startswith("--device=") for a in argv)


def main(argv=None) -> None:
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    if not _device_in(argv):
        argv = argv + ["--device", str(accelerator.device())]
    print(f"eval_rift_dataset_model_free_pvc.py: {accelerator.describe()}", flush=True)
    _model_free.main(argv)


if __name__ == "__main__":
    main()
