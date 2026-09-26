#!/usr/bin/env python
"""CUDA kernel and native-checkpoint validation for the official SH-SAS smoke."""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def check_kernel() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("official SH-SAS CUDA kernel check requires a visible CUDA device")
    device = torch.device("cuda")
    print(f"torch_version={torch.__version__}")
    print(f"gpu_name={torch.cuda.get_device_name(device)}")
    print(f"gpu_capability={torch.cuda.get_device_capability(device)}")

    from eval_sh import EvalSH
    from rift.sh_sas import real_sh_basis_for_directions

    points = torch.tensor(
        [[0.0, 0.0, 0.1], [0.1, 0.05, 0.15], [-0.08, 0.02, 0.18],
         [0.07, -0.04, 0.05], [-0.01, 0.09, 0.02], [0.05, -0.05, 0.2]],
        dtype=torch.float32,
    )
    rx = torch.tensor([[0.0, -0.85, 0.075]], dtype=torch.float32)
    coefficients = torch.arange(6 * 16 * 2, dtype=torch.float32).reshape(6, 16, 2) / 200.0 - 0.4
    output_gradient = torch.arange(1, 13, dtype=torch.float32).reshape(6, 2) / 10.0

    for degree in range(4):
        coefficient_cuda = coefficients.cuda().clone().detach().requires_grad_(True)
        native_output = EvalSH()(points.cuda(), degree, 16, rx.cuda(), coefficient_cuda.reshape(-1, 2))
        basis = real_sh_basis_for_directions(points - rx, degree)
        active_coefficients = coefficients[:, : (degree + 1) ** 2, :]
        expected_output = torch.einsum("nm,nmc->nc", basis, active_coefficients)
        torch.testing.assert_close(native_output.cpu(), expected_output, rtol=1e-5, atol=1e-5)
        (native_output * output_gradient.cuda()).sum().backward()
        expected_gradient = torch.zeros_like(coefficients)
        expected_gradient[:, : (degree + 1) ** 2, :] = basis[:, :, None] * output_gradient[:, None, :]
        torch.cuda.synchronize()
        if not torch.isfinite(native_output).all().item():
            raise AssertionError(f"degree {degree} native output is non-finite")
        if coefficient_cuda.grad is None or not torch.isfinite(coefficient_cuda.grad).all().item():
            raise AssertionError(f"degree {degree} native gradient is non-finite")
        torch.testing.assert_close(
            coefficient_cuda.grad.cpu().reshape(6, 16, 2),
            expected_gradient,
            rtol=1e-5,
            atol=1e-5,
        )
        print(f"degree {degree} PASS")
    print("OFFICIAL_EVAL_SH_CUDA_PASS")


_LOSS_PATTERN = re.compile(
    r"Count\s+(?P<count>\d+): Weight loss\s+(?P<weight>[^,\s]+), "
    r"Sparsity loss\s+(?P<sparsity>[^,\s]+), Smooth loss\s+(?P<smooth>[^,\s]+), "
    r"TV loss\s+(?P<tv>[^,\s]+), Phase loss\s+(?P<phase>[^,\s]+), "
    r"Weight loss\s+(?P<weight_repeat>[^,\s]+)"
)


def check_checkpoints(run: Path, cache_path: str) -> dict:
    first_path = run / "models" / "000000.tar"
    last_path = run / "models" / "000005.tar"
    first = torch.load(first_path, map_location="cpu", weights_only=False)
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    if first.get("global_step") != 0 or last.get("global_step") != 5:
        raise AssertionError("native smoke checkpoints must have global steps 0 and 5")

    first_model = first["network_fn_state_dict"]
    last_model = last["network_fn_state_dict"]
    if set(first_model) != set(last_model):
        raise AssertionError("checkpoint model state keys differ")
    for key in first_model:
        if first_model[key].shape != last_model[key].shape:
            raise AssertionError(f"checkpoint tensor shape differs for {key}")
        for value in (first_model[key], last_model[key]):
            if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all().item():
                raise AssertionError(f"checkpoint tensor is non-finite for {key}")
    parameters_changed = any(not torch.equal(first_model[key], last_model[key]) for key in first_model)
    if not parameters_changed:
        raise AssertionError("native smoke checkpoints show no parameter update")

    dc_export = np.load(run / "numpy" / "comp_albedo0.npy")
    finite_dc_export = bool(np.isfinite(dc_export).all())
    if not finite_dc_export:
        raise AssertionError("native DC export is non-finite")

    from rift.sas_dataset import atomic_json, load_sas_cache

    cache = load_sas_cache(cache_path)
    expected_source_ids = np.asarray(cache.source_ids[cache.train_indices])
    observed_source_ids = np.load(run / "training_source_ids.npy")
    train_source_ids_match = bool(np.array_equal(observed_source_ids, expected_source_ids))
    if not train_source_ids_match:
        raise AssertionError("training source IDs do not match cache train_indices")

    log_text = (run / "native_train.log").read_text(encoding="utf-8", errors="replace")
    records = []
    for match in _LOSS_PATTERN.finditer(log_text):
        record = {
            "count": int(match.group("count")),
            "weight_loss": float(match.group("weight")),
            "sparsity_loss": float(match.group("sparsity")),
            "smooth_loss": float(match.group("smooth")),
            "tv_loss": float(match.group("tv")),
            "phase_loss": float(match.group("phase")),
            "weight_loss_repeat": float(match.group("weight_repeat")),
        }
        if not all(math.isfinite(value) for key, value in record.items() if key != "count"):
            raise AssertionError(f"non-finite native loss record at count {record['count']}")
        records.append(record)
    if not set(range(6)).issubset({record["count"] for record in records}):
        raise AssertionError("native log lacks one or more Count 0..5 records")

    report = {
        "done": True,
        "target_iterations": 6,
        "checkpoint_steps": [0, 5],
        "finite_model_tensors": True,
        "finite_dc_export": True,
        "parameters_changed": True,
        "train_source_ids_match": True,
        "native_loss_records": records,
        "geometry_quality_evaluated": False,
        "heldout_quality_evaluated": False,
    }
    atomic_json(report, run / "official_smoke_validation.json")
    print("OFFICIAL_SHSAS_NATIVE_SMOKE_PASS")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("kernel")
    checkpoints = subparsers.add_parser("checkpoints")
    checkpoints.add_argument("--run", required=True, type=Path)
    checkpoints.add_argument("--cache", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "kernel":
        check_kernel()
    else:
        check_checkpoints(args.run, args.cache)


if __name__ == "__main__":
    main()
