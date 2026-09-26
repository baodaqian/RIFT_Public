#!/usr/bin/env python3
"""Independent Package E review probes; writes synthetic evidence only.

Run on an allocated PVC with --device xpu. This does not repair the package or
relax its gates. Exit 2 means a reproduced audit finding, not a runtime pass.
No real radar responses or source-content hashes are read.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import traceback
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]


def equal_state(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_state(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(equal_state(x, y) for x, y in zip(a, b))
    return a == b


def recovery_and_identity(output, device):
    from test_gotcha_dataset import write_shard, tiny_region
    from rift.gotcha_dataset import GOTCHADataset, sector_split
    from rift_pvc import radar_fields_gotcha as rf
    from rift_pvc import tcnn_torch
    from rift_pvc.radar_fields_training import install

    install()
    inputs = output / "synthetic_input"
    write_shard(inputs / "New_Transfer/shards/pass1_hh.npz", nf=33)
    dataset = GOTCHADataset(inputs, passes=(1,), polarizations=("hh",), region=tiny_region(), num_train=10)
    config = dict(profile="audited-v2", model_backend="upstream-tcnn-torchshim", steps=4,
                  view_batch=1, seed=7, ray_samples=8, eval_every=2, checkpoint_every=1,
                  hidden_dim=16, feature_dim=4, hash_levels=2, hash_final_resolution=8)
    # A fixture-only engineering recipe; production budgets are never modified.
    full_dir, resumed_dir = output / "continuous", output / "interrupted"
    with patch.dict(os.environ, {"RIFT_PVC_TCNN_WEIGHT_GRAD": "bmm:256"}):
        full_result = rf.run_gotcha(dataset=dataset, output_dir=full_dir, config=config, device=device)
        original_step = torch.optim.Adam.step
        calls = [0]

        def stop_at_pending_validation(optimizer, *args, **kwargs):
            result = original_step(optimizer, *args, **kwargs)
            calls[0] += 1
            if calls[0] == 2:
                signal.raise_signal(signal.SIGTERM)
            return result

        with patch.object(torch.optim.Adam, "step", stop_at_pending_validation):
            partial_result = rf.run_gotcha(dataset=dataset, output_dir=resumed_dir, config=config, device=device)
        path = resumed_dir / "checkpoint_latest.pt"
        partial = torch.load(path, map_location="cpu", weights_only=False)
        assert partial_result["status"] == "interrupted" and partial["step"] == partial["pending_validation"] == 2
        old_recipe = copy.deepcopy(partial["recipe"])
        old_identity = copy.deepcopy(partial["tcnn_shim"])
        resumed_result = rf.run_gotcha(dataset=dataset, output_dir=resumed_dir, config=config, device=device, resume=path)
        full = torch.load(full_dir / "checkpoint_final.pt", map_location="cpu", weights_only=False)
        resumed = torch.load(resumed_dir / "checkpoint_final.pt", map_location="cpu", weights_only=False)
        keys = ("model_state_dict", "optimizer", "scheduler", "training_view_coverage", "history",
                "best_val", "training_statistics", "rng_numpy", "rng_python", "rng_torch", "rng_xpu")
        comparison = {key: equal_state(full[key], resumed[key]) for key in keys}
        assert all(comparison.values()), comparison
        assert full_result["status"] == resumed_result["status"] == "complete"
        # The real role-restricted loader still refuses reserved-test requests.
        try:
            list(dataset.observations(1, sector_split()["test"][0], "hh"))
        except PermissionError:
            sealed = True
        else:
            sealed = False
        assert sealed

    with patch.dict(os.environ, {"RIFT_PVC_TCNN_WEIGHT_GRAD": "gemm"}):
        new_recipe = rf.recipe_from_config(config, dataset.region.half_extent_m, len(dataset.viewpoints("train")))
        new_identity = tcnn_torch.identity()
        try:
            rf.validate_resume(partial, dataset, new_recipe, device)
        except ValueError as exc:
            accepted, rejection = False, str(exc)
        else:
            accepted, rejection = True, None
    return {
        "status": "finding" if accepted else "pass", "device": str(device), "synthetic_only": True,
        "continuous_vs_sigterm_resume": comparison, "pending_validation_recovered": True,
        "reserved_test_denied": sealed, "old_weight_grad": old_identity["weight_grad"],
        "new_weight_grad": new_identity["weight_grad"], "recipe_unchanged": old_recipe == new_recipe,
        "changed_reduction_resume_accepted": accepted, "rejection": rejection,
    }


def launcher_failure_probe(output):
    """Exercise the actual B787 shell control flow with stand-ins, no training."""
    fixture = output / "shell_fixture"
    fixture.mkdir()
    bindir, ckdir = fixture / "bin", fixture / "checkpoint"
    bindir.mkdir(); ckdir.mkdir()
    (ckdir / "checkpoint_latest.pth.tar").write_text("stand-in; no checkpoint progress")
    trainer = fixture / "trainer.sh"
    trainer.write_text("#!/bin/bash\nprintf 'Aten Op fallback from XPU to CPU: audit injected warning\\n'\n")
    trainer.chmod(0o755)
    mock = bindir / "python"
    mock.write_text("#!" + sys.executable + "\n" + '''import json, os, sys
if len(sys.argv) > 1 and sys.argv[1] == 'scripts_pvc/radar_fields_smoke_report.py':
    os.execv(sys.executable, [sys.executable, os.environ['AUDIT_REPORTER'], *sys.argv[2:]])
elif len(sys.argv) > 1 and sys.argv[1] == '-c':
    code = sys.argv[2]
    if "['command']" in code: print(os.environ['AUDIT_TRAINER'])
    elif "['output_dir']" in code: print(os.environ['AUDIT_CKDIR'])
    else: print('mock XPU preflight')
elif len(sys.argv) > 1 and sys.argv[1] == '-':
    sys.stdin.read()
    print('mock checkpoint: unchanged step 1, complete=False')
else:
    print(json.dumps({'command': [os.environ['AUDIT_TRAINER']], 'output_dir': os.environ['AUDIT_CKDIR']}))
''')
    mock.chmod(0o755)
    text = (ROOT / "scripts_pvc/smoke_radar_fields_b787_pvc.sbatch").read_text()
    text = text.replace("cd /home/u.db364833/RIFT", 'cd "$AUDIT_FIXTURE"')
    text = text.replace("source .local-setup/activate-pvc.sh", ": # activation replaced only in this shell fixture")
    text = "\n".join('LOG="$AUDIT_LOG"' if line.startswith("LOG=") else line for line in text.splitlines()) + "\n"
    shell = fixture / "smoke-under-review.sh"
    shell.write_text(text)
    log = fixture / "fallback.log"
    log.write_text("Aten Op fallback from XPU to CPU: audit injected warning\n")
    env = {**os.environ, "PATH": str(bindir) + os.pathsep + os.environ["PATH"],
           "AUDIT_FIXTURE": str(fixture), "AUDIT_CKDIR": str(ckdir), "AUDIT_LOG": str(log),
           "AUDIT_TRAINER": str(trainer), "AUDIT_REPORTER": str(ROOT / "scripts_pvc/radar_fields_smoke_report.py"),
           "PVC_OUTPUT_ROOT": str(fixture / "output"), "PVC_TERM_AFTER": "1", "PVC_RESUME_AFTER": "1",
           "SLURM_JOB_NAME": "synthetic-audit", "SLURM_JOB_ID": "synthetic"}
    result = subprocess.run(["bash", str(shell)], env=env, capture_output=True, text=True, timeout=30)
    (fixture / "result.log").write_text(result.stdout + result.stderr)
    if result.returncode != 0:
        rejection = json.loads((fixture / "output/fresh_synthetic.json").read_text())
        assert rejection["status"] == "failed" and "fallback" in rejection["error"], rejection
    return {"status": "finding" if result.returncode == 0 else "pass", "exit_code": result.returncode,
            "injected_fallback": True, "checkpoint_progress": "none", "log": str(fixture / "result.log")}


def network_input_gradients(device):
    """MLP input derivatives are used to train the upstream hash grid."""
    import train_radar_fields_pvc as entry
    from rift_pvc.radar_fields_training import build_model

    args = entry.parse_args(["--recipe", "source-adapted-v3", "--npz-path", "unused.npz", "--device", str(device)])
    model = build_model(args, device).eval()
    seen = {}
    def save_input(module, inputs, output):
        inputs[0].retain_grad()
        seen["tensor"] = inputs[0]
    hook = model.original.xyz_net[0].register_forward_hook(save_input)
    generator = torch.Generator().manual_seed(123)
    xyz = (torch.rand(256, 3, generator=generator).to(device) - .5) * args.extent
    directions = torch.nn.functional.normalize(torch.randn(256, 3, generator=generator), dim=-1).to(device)
    model(xyz, directions, mask_progress=.8)["rcs"].mean().backward()
    hook.remove()
    gx = float(seen["tensor"].grad.norm())
    gp = float(model.original.encode_xyz.params.grad.norm())
    assert gx > 0 and gp > 0
    return {"status": "pass", "xyz_net_input_requires_grad": seen["tensor"].requires_grad,
            "xyz_net_input_gradient_norm": gx, "hash_grid_gradient_norm": gp,
            "implication": "MLP input-gradient parity affects trainable encoding and upstream networks"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "xpu"), default="xpu")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from rift_pvc import accelerator, tcnn_torch
    device = torch.device(args.device)
    if device.type == "xpu":
        assert accelerator.backend() == "xpu" and torch.xpu.is_available() and torch.xpu.device_count() == 1
    else:
        os.environ["RIFT_PVC_TCNN_SHIM"] = "1"
    report = {"job": os.environ.get("SLURM_JOB_ID"), "environment": accelerator.describe(),
              "shim_identity": tcnn_torch.identity(), "checks": {}}
    for name, operation in (
        ("recovery_and_reduction_identity", lambda: recovery_and_identity(args.output, device)),
        ("smoke_shell_acceptance", lambda: launcher_failure_probe(args.output)),
        ("production_network_input_gradients", lambda: network_input_gradients(device)),
    ):
        try:
            result = operation()
        except Exception:
            result = {"status": "error", "traceback": traceback.format_exc()}
        report["checks"][name] = result
        print(json.dumps({name: result}), flush=True)
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 2 if any(r["status"] != "pass" for r in report["checks"].values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
