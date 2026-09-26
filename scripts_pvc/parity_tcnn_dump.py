#!/usr/bin/env python
"""Dump the real tiny-cuda-nn modules of the source-adapted-v3 RadarField (gate 2).

Runs in the CUDA environment on an H100 with the campaign's built tiny-cuda-nn
(``scripts_pvc/parity_tcnn_h100.sbatch``). Builds the five production modules
exactly as ``OriginalRadarFieldsModel`` does (through the authors' unchanged
``RadarField``), records for each module its config, dimensions,
``param_precision()``/``output_precision()``, ``hyperparams()``, the flat fp32
``params``, a fixed random query batch, the forward output and the gradients
of ``sum(output)`` w.r.t. params and inputs; additionally each encoding as a
``tcnn.Encoding(..., dtype=torch.float32)`` twin with the same params (tight
fp32 reference), and one end-to-end RadarField forward in eval mode with the
model's state dict. ``rift_pvc/tests/test_tcnn_parity.py`` replays the NPZ.

    python scripts_pvc/parity_tcnn_dump.py --output DIR/tcnn_parity_h100_<job>.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
AMP = 64.0
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _enum_name(value) -> str:
    return getattr(value, "name", str(value))


def record_module(name, module, x, store, meta, *, kind, dtype_twin=None):
    native = module.native_tcnn_module
    spec = {
        "kind": kind, "n_input_dims": int(module.n_input_dims), "n_output_dims": int(module.n_output_dims),
        "n_params": int(module.params.numel()), "param_precision": _enum_name(native.param_precision()),
        "output_precision": _enum_name(native.output_precision()), "dtype": str(module.dtype),
        "loss_scale": float(module.loss_scale), "seed": int(module.seed), "hyperparams": native.hyperparams(),
        "config": module.encoding_config if kind == "encoding" else module.network_config,
    }
    if kind == "network":
        spec["padded_output_width"] = int(native.n_output_dims())
    meta["modules"][name] = spec
    meta["reduction"] = "mean"
    store[f"{name}/params"] = module.params.detach().float().cpu().numpy()
    store[f"{name}/input"] = x.detach().cpu().numpy()
    xg = x.detach().clone().requires_grad_(True)
    module.zero_grad(set_to_none=True)
    out = module(xg)
    out.float().mean().backward()   # mean: keeps TCNN's loss-scaled fp16 backward finite
    store[f"{name}/output"] = out.detach().float().cpu().numpy()
    if module.params.numel():
        store[f"{name}/grad_params"] = module.params.grad.detach().float().cpu().numpy()
    if xg.grad is not None:
        store[f"{name}/grad_input"] = xg.grad.detach().float().cpu().numpy()
    if dtype_twin is not None:
        import tinycudann as tcnn
        twin = tcnn.Encoding(n_input_dims=module.n_input_dims, encoding_config=module.encoding_config,
                             dtype=dtype_twin).cuda()
        assert twin.params.numel() == module.params.numel()
        with torch.no_grad():
            twin.params.copy_(module.params.detach().float())
        xt = x.detach().clone().requires_grad_(True)
        out = twin(xt)
        out.float().mean().backward()
        store[f"{name}/fp32/output"] = out.detach().float().cpu().numpy()
        if twin.params.numel():
            store[f"{name}/fp32/grad_params"] = twin.params.grad.detach().float().cpu().numpy()
        if xt.grad is not None:
            store[f"{name}/fp32/grad_input"] = xt.grad.detach().float().cpu().numpy()
        spec["fp32_twin"] = {"param_precision": _enum_name(twin.native_tcnn_module.param_precision()),
                             "output_precision": _enum_name(twin.native_tcnn_module.output_precision())}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, help="NPZ path; a .json and a .state.npz sidecar are written beside it")
    parser.add_argument("--points", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    assert torch.cuda.is_available(), "the parity dump needs the real tiny-cuda-nn on a CUDA device"
    import tinycudann as tcnn
    import train_radar_fields as rf
    from rift.radar_fields_upstream import OriginalRadarFieldsModel, REFERENCE_ROOT
    rf_args = rf.parse_args(["--recipe", "source-adapted-v3", "--npz-path", "unused.npz", "--device", "cuda"])
    assert rf_args.model_backend == "upstream-tcnn"
    torch.manual_seed(args.seed)
    wrapper = OriginalRadarFieldsModel(rf_args).cuda()
    model = wrapper.original
    assert isinstance(model.encode_xyz, tcnn.Encoding) and isinstance(model.xyz_net[0], tcnn.Network)
    store, meta = {}, {
        "job": os.environ.get("SLURM_JOB_ID"), "host": os.uname().nodename, "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "tinycudann": getattr(tcnn, "__version__", "unknown"), "tinycudann_file": getattr(tcnn, "__file__", None),
        "reference_root": str(REFERENCE_ROOT), "points": args.points, "seed": args.seed,
        "rf_args": {k: v for k, v in vars(rf_args).items() if isinstance(v, (int, float, str, bool, type(None)))},
        "modules": {},
        "model": {"radarfield_kwargs": dict(
            in_dim=3, xyz_encoding="HashGrid", num_layers=4, hidden_dim=rf_args.hidden_dim,
            xyz_feat_dim=rf_args.feature_dim, alpha_dim=1, alpha_activation="sigmoid",
            sigmoid_tightness=rf_args.sigmoid_tightness, rd_dim=1, softplus_rd=True, angle_dim=3, angle_in_layer=3,
            angle_encoding="SphericalHarmonics", resolution=rf_args.hash_final_resolution,
            n_levels=rf_args.hash_levels, bound=1, bn=not rf_args.no_batch_norm, use_tcnn=True), "sin_epoch": 0.8},
    }
    n = args.points
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    xyz = torch.rand(n, 3, generator=g).cuda()
    angle = torch.nn.functional.normalize(torch.randn(n, 3, generator=g), dim=-1).cuda()
    record_module("encode_xyz", model.encode_xyz, xyz, store, meta, kind="encoding", dtype_twin=torch.float32)
    record_module("encode_angle", model.encode_angle, angle, store, meta, kind="encoding", dtype_twin=torch.float32)
    # Network inputs at the scale the release produces: encoded features and the SH/feature concatenation.
    with torch.no_grad():
        feats = model.encode_xyz(xyz).float()
        xyz_features = model.xyz_net(feats).float()
        angle_encoded = model.encode_angle(angle).float()
    record_module("xyz_net", model.xyz_net[0], feats, store, meta, kind="network")
    record_module("alpha_net", model.alpha_net, xyz_features, store, meta, kind="network")
    record_module("rd_net", model.rd_net, torch.cat([angle_encoded, xyz_features], dim=-1), store, meta, kind="network")
    # Amplified set: the initial grid features (|f| <= 1e-4) are fp16 subnormals; scale the grid by AMP so the
    # MLP comparison also covers trained-like feature magnitudes (fp16 normal range).
    meta["amp_factor"] = AMP
    with torch.no_grad():
        model.encode_xyz.params.mul_(AMP)
    record_module("encode_xyz@amp", model.encode_xyz, xyz, store, meta, kind="encoding", dtype_twin=torch.float32)
    with torch.no_grad():
        feats = model.encode_xyz(xyz).float()
        xyz_features = model.xyz_net(feats).float()
    record_module("xyz_net@amp", model.xyz_net[0], feats, store, meta, kind="network")
    record_module("alpha_net@amp", model.alpha_net, xyz_features, store, meta, kind="network")
    record_module("rd_net@amp", model.rd_net, torch.cat([angle_encoded, xyz_features], dim=-1), store, meta, kind="network")
    model.eval()
    with torch.no_grad():
        out = model(xyz, angle, sin_epoch=meta["model"]["sin_epoch"])
    store["model@amp/alpha"], store["model@amp/rd"] = out["alpha"].float().cpu().numpy(), out["rd"].float().cpu().numpy()
    with torch.no_grad():
        model.encode_xyz.params.div_(AMP)
    model.eval()
    with torch.no_grad():
        out = model(xyz, angle, sin_epoch=meta["model"]["sin_epoch"])
    store["model/xyz"], store["model/angle"] = xyz.cpu().numpy(), angle.cpu().numpy()
    store["model/alpha"], store["model/rd"] = out["alpha"].float().cpu().numpy(), out["rd"].float().cpu().numpy()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **store)
    np.savez(str(output)[:-4] + ".state.npz", **{k: v.detach().float().cpu().numpy() if v.is_floating_point() else v.cpu().numpy()
                                                 for k, v in model.state_dict().items()})
    Path(str(output)[:-4] + ".json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    print(json.dumps({k: {"n_params": v["n_params"], "param_precision": v["param_precision"],
                          "output_precision": v["output_precision"], "n_output_dims": v["n_output_dims"]}
                      for k, v in meta["modules"].items()}, indent=2))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
