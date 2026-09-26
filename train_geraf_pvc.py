#!/usr/bin/env python
"""PVC (Intel XPU) entry point for the GeRaF baseline.

Package B of ``RIFT_PVC_Adaptation.md``. Same CLI as ``train_geraf.py``, so the
PVC dataset frontends can substitute the script name and nothing else:

    train_rift_dataset_pvc.py --method geraf   ->  train_geraf_pvc.py --implementation source_v1
                                                   --npz-path ... --role-manifest ...
                                                   --cache-root ... --checkpoint-dir ...
                                                   --resume|--no-resume [--resume-path P]
                                                   --source-config protocols/geraf_mf48_1t1r.json
    train_gotcha_dataset_pvc.py --method geraf ->  run_gotcha(dataset=, output_dir=, config=,
                                                             device=, resume=)

Routing:

* ``--implementation source_v1`` (the production recipe, and the default) runs
  entirely inside ``rift_pvc``: ``geraf_source_cli`` -> ``geraf_source_training``
  -> ``geraf_source`` -> ``rift_pvc.vendor.geraf_sens.rf_rendering``.
* ``hardened_v1`` / ``legacy`` are the compatibility recipes. They run the
  unchanged ``train_geraf`` module with only its three accelerator-specific
  helpers rebound to their ``rift_pvc.geraf_rng`` twins:

      _seed_everything   -> geraf_rng.seed_everything   (cudnn flags kept, CUDA-guarded)
      _capture_rng_state -> geraf_rng.capture_rng_state (torch_cuda kept, torch_xpu added)
      _restore_rng_state -> geraf_rng.restore_rng_state (restores by active backend)

  The rebinding happens in this process only; ``train_geraf.py`` on disk and the
  CUDA pipeline are untouched.

``train_rift_dataset*.py`` never passes ``--device`` for GeRaF, so the device
comes from the ``--device`` default: ``rift_pvc.geraf_source_cli`` resolves it
through the shim, and on the compatibility path this entry point injects
``--device <accelerator.device()>`` when the caller omits the flag. An explicit
``--device`` (including ``xpu``) is always honoured.

By default this entry point refuses to run without an XPU device, matching
``train_pvc.py``; set ``RIFT_PVC_ALLOW_BACKEND=cpu`` (tests) or ``=cuda`` to
override.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rift_pvc import accelerator  # noqa: E402
from rift_pvc import geraf_rng  # noqa: E402
import train_geraf as _geraf  # noqa: E402  (the unchanged trainer)


# Literal capability declaration read by train_gotcha_dataset*.py via ast.literal_eval.
# Identical to train_geraf.GOTCHA_BACKEND; asserted against it in the PVC tests.
GOTCHA_BACKEND = {
    "schema": "rift_gotcha_backend_v1", "method": "geraf", "callable": "run_gotcha",
    "selection_unit": "pass_sector", "joint_passes": True,
    "native_frequency_policy": "ragged_exact", "polarizations": ["hh", "hv", "vh", "vv"],
    "metric_domain": "native_coherent_mf_magnitude",
    "implementation": "source_v1",
}

REBIND = {
    "_seed_everything": geraf_rng.seed_everything,
    "_capture_rng_state": geraf_rng.capture_rng_state,
    "_restore_rng_state": geraf_rng.restore_rng_state,
}


def run_gotcha(*, dataset, output_dir, config, device, resume):
    from rift_pvc.geraf_gotcha import run_gotcha as backend
    return backend(dataset=dataset, output_dir=output_dir, config=config, device=device, resume=resume)


def install():
    """Rebind the accelerator-specific names in the ``train_geraf`` namespace."""
    missing = [name for name in REBIND if not callable(getattr(_geraf, name, None))]
    if missing:
        raise RuntimeError(f"train_geraf.py no longer defines {missing}; train_geraf_pvc.py must be revisited")
    for name, twin in REBIND.items():
        setattr(_geraf, name, twin)
    return _geraf


def check_backend():
    allowed = {"xpu"} | {b.strip() for b in os.environ.get("RIFT_PVC_ALLOW_BACKEND", "").split(",") if b.strip()}
    backend = accelerator.backend()
    if backend not in allowed:
        raise RuntimeError(
            f"train_geraf_pvc.py is the PVC entry point and found backend {backend!r}; "
            "use train_geraf.py on CUDA, or set RIFT_PVC_ALLOW_BACKEND=cpu|cuda to override")
    return backend


def _device_in(argv) -> bool:
    return any(a == "--device" or a.startswith("--device=") for a in argv)


def parse_args(argv=None):
    """``train_geraf.parse_args`` with the PVC source CLI and device default."""
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    import argparse
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--implementation", choices=("source_v1", "hardened_v1", "legacy"), default="source_v1")
    selected, _ = selector.parse_known_args(argv)
    if selected.implementation == "source_v1":
        from rift_pvc.geraf_source_cli import parse_args as source_args
        return source_args(argv)
    if not _device_in(argv):
        # The compatibility parser's default is "cuda" or "cpu"; neither names
        # the PVC card. Injecting the flag keeps the parser itself unchanged.
        argv = argv + ["--device", str(accelerator.device())]
    return _geraf.parse_compatibility_args(argv)


def main(argv=None):
    backend = check_backend()
    args = parse_args(argv)
    print(f"train_geraf_pvc.py: {accelerator.describe()}", flush=True)
    if backend == "xpu":
        print("train_geraf_pvc.py: PYTORCH_DEBUG_XPU_FALLBACK="
              f"{os.environ.get('PYTORCH_DEBUG_XPU_FALLBACK', 'unset')} SYCL_CACHE_PERSISTENT="
              f"{os.environ.get('SYCL_CACHE_PERSISTENT', 'unset')}", flush=True)
    if getattr(args, "implementation", "legacy") == "source_v1":
        from rift_pvc.geraf_source_cli import run
        return run(args)
    install()
    _geraf.parse_args = lambda _argv=None: args
    return _geraf.main()


if __name__ == "__main__":
    # Same convention as train_geraf.py: main() is called for its effects, not
    # for an exit status. A source_v1 run returns the trainer's result dict, and
    # `raise SystemExit(<dict>)` would make a successful run exit 1. The one
    # nonzero status this program produces is geraf_source_cli.run's
    # SystemExit(143) for a clean interruption, which propagates by itself.
    main()
