#!/usr/bin/env python3
"""Independent Package F evidence collection; no production model or recipe edits.

Replay both saved SH passes even when an earlier gate fails. Trace the first
projection inputs and the six pre-filter products to locate end-to-end drift.
Only the existing H100 fixture is read; no dataset responses are accessed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch


def compare(a, b):
    a, b = a.detach().cpu(), b.detach().cpu()
    delta = (a - b).abs()
    scale = float(b.abs().max())
    return {
        "max_abs": float(delta.max()),
        "ref_max_abs": scale,
        "relative_max": float(delta.max()) / max(scale, 1e-12),
        "relative_l2": float(torch.linalg.vector_norm((a - b).double()))
        / max(float(torch.linalg.vector_norm(b.double())), 1e-30),
        "pixel_gate": bool(((delta <= 1e-6) | (delta <= 1e-4 * b.abs())).all()),
        "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "xpu"), default="xpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from rift_pvc import accelerator
    from rift_pvc.radarsplat_xpu_backend import load_xpu_reference, backend_record
    from rift_pvc.radarsplat_release import ReleasedRenderer, release_loss
    from rift.radarsplat_b7873200 import RadarSplatGrid
    if args.device == "xpu":
        assert accelerator.backend() == "xpu" and torch.xpu.is_available()
        assert not torch.cuda.is_available()
    payload = torch.load(args.dump, map_location="cpu", weights_only=False)
    assert payload["schema"] == "rift_pvc_radarsplat_parity_dump_v1"
    assert payload["environment"]["cuda"] and "H100" in payload["environment"]["gpu"]
    rendering, ssim = load_xpu_reference(device=args.device)
    import gsplat.cuda._torch_impl_radar as radar_impl
    view = payload["view"]
    grid = RadarSplatGrid(**view["grid"])
    report = {"job": os.environ.get("SLURM_JOB_ID"), "reference": str(args.dump),
              "reference_environment": payload["environment"],
              "environment": accelerator.describe(), "backend": backend_record(),
              "loaded_modules": os.environ.get("LOADEDMODULES", ""),
              "fixture_shN_nonzero": int(torch.count_nonzero(payload["scene"]["splats"]["shN"])),
              "passes": {}}
    passed = True
    for saved in payload["passes"]:
        pass_id = saved["pass_id"]
        projection_ref = next(c for c in payload["calls"]
                              if c["pass_id"] == pass_id and c["op"] == "fully_fused_projection")
        products_ref = [c for c in payload["calls"]
                        if c["pass_id"] == pass_id and c["op"] == "_rasterize_to_radar_pixels"]
        stats = {"projection_inputs": {}, "products_before_filter": []}
        original_project = rendering.fully_fused_projection
        original_raster = radar_impl._rasterize_to_radar_pixels

        def project(*values, **kwargs):
            for i, (a, b) in enumerate(zip(values, projection_ref["args"])):
                if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                    stats["projection_inputs"][str(i)] = compare(a, b)
            return original_project(*values, **kwargs)

        def raster(*values, **kwargs):
            result = original_raster(*values, **kwargs)
            index = len(stats["products_before_filter"])
            stats["products_before_filter"].append(compare(result, products_ref[index]["outputs"]))
            return result

        rendering.fully_fused_projection = project
        radar_impl._rasterize_to_radar_pixels = raster
        started = time.monotonic()
        try:
            splats = torch.nn.ParameterDict({k: torch.nn.Parameter(v.to(args.device).clone())
                                            for k, v in payload["scene"]["splats"].items()})
            renderer = ReleasedRenderer(rendering, view["units_per_m"], local_azimuth=view["local_azimuth"])
            power, occupancy = renderer(splats, view["pose"].to(args.device), grid,
                                        saved["active_degree"], view["background"].to(args.device))
            losses = release_loss(power, occupancy, view["target_masked"].to(args.device),
                                  view["labels"].to(args.device), splats, ssim)
            losses["total"].backward()
            stats["power"] = compare(power, saved["power"])
            stats["occupancy"] = compare(occupancy, saved["occupancy"])
            stats["losses"] = {k: {"actual": float(v.detach()), "reference": saved["losses"][k],
                                   "relative": abs(float(v.detach()) - saved["losses"][k])
                                   / max(abs(saved["losses"][k]), 1e-12)} for k, v in losses.items()}
            stats["gradients"] = {k: compare(v.grad, saved["grads"][k]) for k, v in splats.items()}
            stats["seconds"] = time.monotonic() - started
            stats["declared_gates_pass"] = (
                stats["power"]["pixel_gate"] and stats["occupancy"]["pixel_gate"]
                and all(x["relative"] <= 1e-5 for x in stats["losses"].values())
                and all(x["finite"] and x["relative_max"] <= 1e-3 for x in stats["gradients"].values()))
            passed &= stats["declared_gates_pass"]
            report["passes"][pass_id] = stats
            print(json.dumps({"pass": pass_id, **stats}), flush=True)
        finally:
            rendering.fully_fused_projection = original_project
            radar_impl._rasterize_to_radar_pixels = original_raster
    report["declared_gates_pass"] = passed
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
