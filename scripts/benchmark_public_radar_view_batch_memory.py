#!/usr/bin/env python
"""Measure full-domain PublicRadar aligned-view training memory on one GPU."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "rift.public_radar.view_batch_memory_v1"
MANAGEABLE_CUDA_PEAK_BYTES = 12 * 1024**3
MANAGEABLE_HOST_PEAK_KIB = 24 * 1024**2


def load_trainer():
    path = PROJECT_ROOT / "scripts" / "public_radar_gotcha_full_domain_v1_partial.py"
    spec = importlib.util.spec_from_file_location(
        "rift_public_radar_view_batch_memory_trainer", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def process_memory_kib():
    values = {}
    with Path("/proc/self/status").open(encoding="utf-8") as handle:
        for line in handle:
            name, _, value = line.partition(":")
            if name in ("VmRSS", "VmHWM"):
                values[name] = int(value.strip().split()[0])
    values["ru_maxrss"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return values


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def cuda_equivalence_gate(trainer, device, batch_size):
    """Compare aligned CUDA rendering and gradients with scalar view calls."""
    torch.manual_seed(2900 + batch_size)
    model = trainer.PLANAR.FixedPlanarSHScene(
        5, 4, 0.4, 3, device, init_scale=0.02
    )
    dtheta = torch.linspace(0.25, 0.75, batch_size, device=device)
    dphi = torch.linspace(0.4, 1.6, batch_size, device=device)
    offsets = torch.linspace(
        -0.3, 0.3, batch_size, dtype=torch.float64, device=device
    )
    platforms = torch.stack(
        (7.0 + offsets, 4.0 - 0.5 * offsets, 3.0 + 0.25 * offsets), dim=-1
    )
    frequencies = torch.linspace(
        9.0e9, 9.2e9, 17, dtype=torch.float64, device=device
    )
    kvector = trainer.get_kvector(frequencies, trainer.cc)
    kwargs = {
        "phase_sign": -1.0,
        "freq_indices": torch.arange(frequencies.numel(), device=device),
        "compute_dtype": torch.float64,
        "range_model": "none",
        "propagation_model": "monostatic_near_field_reference",
        "reference_range_m": 7.5,
        "scene_center_m": (0.0, 0.0, 0.0),
    }

    def aligned_column_chunks(index):
        def factory():
            for positions, weights in model.scatterer_view_chunks(
                dtheta, dphi, 3
            ):
                yield positions, weights[:, index]

        return factory

    def scalar_prediction():
        values = []
        for index in range(batch_size):
            platform = platforms[index].reshape(1, 3)
            value = trainer.SERIAL.range_forward_operator_chunks(
                frequencies,
                kvector,
                platform,
                platform,
                aligned_column_chunks(index),
                pair_chunk=1,
                **kwargs,
            )
            values.append(value[:, 0, 0])
        return torch.stack(values)

    reference = scalar_prediction()
    batched = trainer.SERIAL.range_forward_operator_aligned_view_chunks(
        frequencies,
        kvector,
        platforms,
        platforms,
        lambda: model.scatterer_view_chunks(dtheta, dphi, 3),
        **kwargs,
    )
    forward_relative_error = float(
        torch.linalg.vector_norm(batched - reference)
        / torch.linalg.vector_norm(reference).clamp_min(1.0e-30)
    )
    target = torch.complex(
        torch.randn_like(reference.real), torch.randn_like(reference.real)
    )
    model.zero_grad(set_to_none=True)
    reference_loss = (reference - target).abs().square().sum()
    reference_loss.backward()
    reference_gradients = [
        parameter.grad.detach().clone() for parameter in model.parameters()
    ]
    model.zero_grad(set_to_none=True)
    batched = trainer.SERIAL.range_forward_operator_aligned_view_chunks(
        frequencies,
        kvector,
        platforms,
        platforms,
        lambda: model.scatterer_view_chunks(dtheta, dphi, 3),
        **kwargs,
    )
    batched_loss = (batched - target).abs().square().sum()
    batched_loss.backward()
    gradient_relative_error = max(
        float(
            torch.linalg.vector_norm(parameter.grad - expected)
            / torch.linalg.vector_norm(expected).clamp_min(1.0e-30)
        )
        for parameter, expected in zip(model.parameters(), reference_gradients)
    )
    loss_relative_error = abs(float(batched_loss - reference_loss)) / max(
        abs(float(reference_loss)), 1.0e-30
    )
    if forward_relative_error > 1.0e-8:
        raise RuntimeError(
            f"CUDA aligned forward relative error {forward_relative_error:.9g}"
        )
    if loss_relative_error > 1.0e-8:
        raise RuntimeError(
            f"CUDA aligned loss relative error {loss_relative_error:.9g}"
        )
    if gradient_relative_error > 1.0e-7:
        raise RuntimeError(
            f"CUDA aligned gradient relative error {gradient_relative_error:.9g}"
        )
    del (
        model,
        reference,
        batched,
        target,
        reference_loss,
        batched_loss,
        reference_gradients,
        dtheta,
        dphi,
        platforms,
        frequencies,
        kvector,
        kwargs,
    )
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return {
        "cuda_equivalence_forward_relative_error": forward_relative_error,
        "cuda_equivalence_loss_relative_error": loss_relative_error,
        "cuda_equivalence_gradient_relative_error": gradient_relative_error,
    }


def one_step(model, optimizer, renderer, items):
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    start = time.perf_counter()
    predicted, measured = renderer.raw_prediction_and_measurement_batch(model, items)
    torch.cuda.synchronize()
    forward_seconds = time.perf_counter() - start
    target_power = measured.abs().square().mean().clamp_min(1.0e-30)
    loss = (predicted - measured).abs().square().mean() / target_power
    start = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize()
    backward_seconds = time.perf_counter() - start
    start = time.perf_counter()
    optimizer.step()
    torch.cuda.synchronize()
    optimizer_seconds = time.perf_counter() - start
    return loss, forward_seconds, backward_seconds, optimizer_seconds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz-path", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--batch-size", required=True, type=int, choices=(1, 2, 4, 8))
    parser.add_argument("--point-chunk", type=int, default=32768)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("view-batch memory benchmark requires CUDA")
    if args.point_chunk <= 0:
        raise ValueError("point_chunk must be positive")

    result_path = Path(args.run_root).resolve() / f"gotcha_deg3_b{args.batch_size}.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite existing benchmark: {result_path}")

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda:0")
    trainer = load_trainer()
    equivalence = cuda_equivalence_gate(trainer, device, args.batch_size)
    arrays, metadata, partition, _ = trainer.load_dataset_contract(
        args.npz_path, "gotcha_p2_full_domain"
    )
    train_indices = np.asarray(arrays["train_indices"], dtype=np.int64)
    train_dataset = trainer.PecSphereNPZDataset(arrays, train_indices)
    if len(train_dataset) < args.batch_size + 1:
        raise ValueError("training dataset is too small for the requested benchmark")
    model = trainer.make_scene("gotcha_p2_full_domain", 3, device)
    renderer = trainer.SerializedRenderer(
        arrays, metadata, device, args.point_chunk, pair_chunk=1
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3.0e-5, eps=1.0e-15, weight_decay=0.0
    )
    warmup_items = [train_dataset[0]]
    target_items = [train_dataset[index] for index in range(1, args.batch_size + 1)]
    common = {
        "schema": SCHEMA,
        "batch_size": int(args.batch_size),
        "point_chunk": int(args.point_chunk),
        "degree": 3,
        "points": int(model.n_points),
        "frequencies": int(renderer.freqs.numel()),
        "train_views": int(partition["train"]),
        "device": torch.cuda.get_device_name(device),
        "cuda_total_bytes": int(torch.cuda.get_device_properties(device).total_memory),
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        **equivalence,
    }

    try:
        warmup_loss, _, _, _ = one_step(
            model, optimizer, renderer, warmup_items
        )
        if not math.isfinite(float(warmup_loss.detach())):
            raise RuntimeError("warmup loss is non-finite")
        del warmup_loss
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        free_before, total_before = torch.cuda.mem_get_info(device)
        baseline_allocated = int(torch.cuda.memory_allocated(device))
        baseline_reserved = int(torch.cuda.memory_reserved(device))
        host_before = process_memory_kib()
        torch.cuda.reset_peak_memory_stats(device)

        loss, forward_seconds, backward_seconds, optimizer_seconds = one_step(
            model, optimizer, renderer, target_items
        )
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        free_after, total_after = torch.cuda.mem_get_info(device)
        host_after = process_memory_kib()
        gradients_finite = all(
            parameter.grad is not None
            and bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        payload = {
            **common,
            "status": "pass",
            "loss": float(loss.detach()),
            "loss_finite": math.isfinite(float(loss.detach())),
            "gradients_finite": bool(gradients_finite),
            "forward_seconds": float(forward_seconds),
            "backward_seconds": float(backward_seconds),
            "optimizer_seconds": float(optimizer_seconds),
            "cuda_baseline_allocated_bytes": baseline_allocated,
            "cuda_baseline_reserved_bytes": baseline_reserved,
            "cuda_peak_allocated_bytes": peak_allocated,
            "cuda_peak_reserved_bytes": peak_reserved,
            "cuda_peak_reserved_fraction_total": float(peak_reserved / total_after),
            "cuda_headroom_at_peak_bytes": int(total_after - peak_reserved),
            "cuda_manageable_peak_limit_bytes": MANAGEABLE_CUDA_PEAK_BYTES,
            "cuda_incremental_peak_allocated_bytes": peak_allocated
            - baseline_allocated,
            "cuda_incremental_peak_reserved_bytes": peak_reserved
            - baseline_reserved,
            "cuda_free_before_bytes": int(free_before),
            "cuda_free_after_bytes": int(free_after),
            "cuda_mem_get_info_total_before_bytes": int(total_before),
            "cuda_mem_get_info_total_after_bytes": int(total_after),
            "host_before_kib": host_before,
            "host_after_kib": host_after,
            "host_rss_delta_kib": int(
                host_after.get("VmRSS", 0) - host_before.get("VmRSS", 0)
            ),
            "host_lifetime_peak_kib": int(host_after.get("VmHWM", 0)),
            "host_manageable_peak_limit_kib": MANAGEABLE_HOST_PEAK_KIB,
        }
        payload["memory_manageable"] = bool(
            peak_reserved <= MANAGEABLE_CUDA_PEAK_BYTES
            and payload["host_lifetime_peak_kib"] <= MANAGEABLE_HOST_PEAK_KIB
        )
        if not payload["loss_finite"] or not payload["gradients_finite"]:
            payload["status"] = "nonfinite"
        elif not payload["memory_manageable"]:
            payload["status"] = "unsafe"
    except torch.cuda.OutOfMemoryError as exc:
        payload = {
            **common,
            "status": "oom",
            "memory_manageable": False,
            "error": str(exc),
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "cuda_manageable_peak_limit_bytes": MANAGEABLE_CUDA_PEAK_BYTES,
            "host_after_kib": process_memory_kib(),
            "host_manageable_peak_limit_kib": MANAGEABLE_HOST_PEAK_KIB,
        }
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

    atomic_json(result_path, payload)
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
