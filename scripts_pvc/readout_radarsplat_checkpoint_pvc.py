#!/usr/bin/env python3
"""PVC twin of ``scripts/readout_radarsplat_checkpoint.py`` (collection RadarSplat readout).

Same CLI and result schema. Importing ``rift_pvc.radarsplat_release_training``
installs the PVC engine twins, so a released-schema checkpoint is read out on
the torch-mirror renderer (``fork_torch_mirror_xpu_v1``) through
``load_xpu_reference`` instead of the CUDA loader, and the report carries the
checkpoint's ``backend.json`` and the readout backend (D5). Legacy/audit_v1
checkpoints use the original independent renderer on the requested device.
``--device`` defaults to the accelerator device.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import rift_pvc.radarsplat_release_training as _engine_pvc  # noqa: E402  (installs the PVC engine twins)
from rift_pvc import accelerator  # noqa: E402
from rift_pvc.radarsplat_xpu_backend import BACKEND_IDENTITY, SSIM_IDENTITY, read_sidecar  # noqa: E402
import scripts.readout_radarsplat_checkpoint as _original  # noqa: E402  (unchanged CLI)


def readout(*, checkpoint_path, cache_root, device=None, role="validation", object_name=None,
            gaussian_chunk_size=64, max_raster_candidate_pairs=2_000_000, geometry_path=None):
    _engine_pvc.install()
    device = accelerator.device() if device is None else device
    result = _original.readout(checkpoint_path=checkpoint_path, cache_root=cache_root, device=device, role=role,
                               object_name=object_name, gaussian_chunk_size=gaussian_chunk_size,
                               max_raster_candidate_pairs=max_raster_candidate_pairs, geometry_path=geometry_path)
    if "pvc" not in result:  # legacy/audit_v1 recipes: the independent torch renderer, no released kernels involved
        sidecar = read_sidecar(Path(checkpoint_path).parent)
        result["pvc"] = dict(readout_backend=f"independent torch renderer on {device}", readout_ssim=None,
                             checkpoint_backend=sidecar if sidecar is not None else "no backend.json beside the checkpoint")
    result["pvc"]["entrypoint"] = "scripts_pvc/readout_radarsplat_checkpoint_pvc.py"
    result["pvc"]["released_backend"] = dict(radarsplat_backend=BACKEND_IDENTITY, ssim=SSIM_IDENTITY)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--object")
    parser.add_argument("--role", choices=("train", "validation"), default="validation")
    parser.add_argument("--device", default=str(accelerator.device()))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--geometry", type=Path)
    parser.add_argument("--gaussian-chunk-size", type=int, default=64)
    parser.add_argument("--max-raster-candidate-pairs", type=int, default=2_000_000)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    result = readout(checkpoint_path=args.checkpoint, cache_root=args.cache_root, device=args.device, role=args.role,
                     object_name=args.object, gaussian_chunk_size=args.gaussian_chunk_size,
                     max_raster_candidate_pairs=args.max_raster_candidate_pairs, geometry_path=args.geometry)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
