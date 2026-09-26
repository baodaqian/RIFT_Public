#!/usr/bin/env python
"""Train the SH-SAS baseline on a RIFT coherent-radar NPZ dataset.

The representation and rendering priors live in :mod:`rift.sh_sas`; this
driver supplies the modality-specific exact bistatic swept-frequency operator,
the seed-42 fixed-tail B787 split, resumable checkpoints, and common coherent
signal metrics.  No STL, point cloud, backprojection image, or geometry label
enters training.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch

from rift.calibration import GlobalComplexGain
from rift.config import cc
from rift.radar_fields_dataset import (
    RadarFieldsArrays,
    build_frequency_grid,
    load_radar_fields_npz,
    split_view_indices,
)
from rift.range_operator import range_forward_operator
from rift.sh_sas import (
    PAPER_HASH_BASE_RESOLUTION,
    PAPER_HASH_FINAL_RESOLUTION,
    PAPER_HASH_LEVELS,
    PAPER_MLP_WIDTH,
    PAPER_SH_DEGREE,
    SHSASField,
)


CONTRACT_VERSION = 1
STOP_REQUESTED = False


def request_stop(signum, _frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(
        f"Received signal {signum}; will publish checkpoint_latest after the current view.",
        flush=True,
    )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_axis_indices(
    rng: np.random.Generator, total: int, requested: int, random: bool
) -> np.ndarray:
    if requested <= 0 or requested >= total:
        return np.arange(total, dtype=np.int64)
    if random:
        return np.sort(rng.choice(total, size=requested, replace=False).astype(np.int64))
    return np.unique(np.linspace(0, total - 1, requested).round().astype(np.int64))


def select_frequency_indices(
    rng: np.random.Generator, total: int, requested: int, random: bool
) -> np.ndarray:
    return select_axis_indices(rng, total, requested, random=random)


def measured_view(
    arrays: RadarFieldsArrays,
    view_index: int,
    tx_indices: np.ndarray,
    rx_indices: np.ndarray,
    freq_indices: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Measured complex cube in the operator's ``[freq,rx,tx]`` layout."""
    cube = arrays.response_view(view_index).mean(axis=2)  # [tx,rx,freq], honors sealed roles
    cube = cube[np.ix_(tx_indices, rx_indices, freq_indices)]
    return torch.as_tensor(cube, dtype=torch.complex64, device=device).permute(2, 1, 0)


def metric_record(predicted: torch.Tensor, measured: torch.Tensor) -> Dict[str, float]:
    measured = measured.to(predicted.dtype)
    diff = predicted - measured
    return {
        "sq_error": float(diff.abs().square().sum().detach()),
        "target_power": float(measured.abs().square().sum().detach()),
        "count": float(diff.numel()),
        "real_l1_sum": float(diff.real.abs().sum().detach()),
        "imag_l1_sum": float(diff.imag.abs().sum().detach()),
        "abs_l1_sum": float((predicted.abs() - measured.abs()).abs().sum().detach()),
        "real_sq_sum": float(diff.real.square().sum().detach()),
        "imag_sq_sum": float(diff.imag.square().sum().detach()),
        "abs_sq_sum": float((predicted.abs() - measured.abs()).square().sum().detach()),
    }


def combine_metrics(records: Iterable[Dict[str, float]]) -> Dict[str, float]:
    records = list(records)
    totals = {
        key: sum(record[key] for record in records)
        for key in (
            "sq_error",
            "target_power",
            "count",
            "real_l1_sum",
            "imag_l1_sum",
            "abs_l1_sum",
            "real_sq_sum",
            "imag_sq_sum",
            "abs_sq_sum",
        )
    }
    count = max(totals["count"], 1.0)
    target_power = max(totals["target_power"], 1.0e-30)
    return {
        "rel_mse": totals["sq_error"] / target_power,
        "l1_real": totals["real_l1_sum"] / count,
        "l1_imag": totals["imag_l1_sum"] / count,
        "l1_abs": totals["abs_l1_sum"] / count,
        "mse_real": totals["real_sq_sum"] / count,
        "mse_imag": totals["imag_sq_sum"] / count,
        "mse_abs": totals["abs_sq_sum"] / count,
        **totals,
    }


def build_model(args, device: torch.device) -> SHSASField:
    return SHSASField(
        extent=args.extent,
        granularity=args.granularity,
        sh_degree=args.sh_degree,
        hidden_dim=args.hidden_dim,
        hash_levels=args.hash_levels,
        hash_features=args.hash_features,
        hash_base_resolution=args.hash_base_resolution,
        hash_final_resolution=args.hash_final_resolution,
        hash_log2_size=args.hash_log2_size,
        device=device,
    ).to(device)


def view_objective(
    model: SHSASField,
    gain: GlobalComplexGain,
    arrays: RadarFieldsArrays,
    frequencies: torch.Tensor,
    kvector: torch.Tensor,
    view_index: int,
    tx_indices_np: np.ndarray,
    rx_indices_np: np.ndarray,
    freq_indices_np: np.ndarray,
    args,
    device: torch.device,
    include_regularizers: bool,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, float]]:
    tx_pos = torch.as_tensor(
        arrays.tx_pos[view_index, tx_indices_np], dtype=torch.float32, device=device
    )
    rx_pos = torch.as_tensor(
        arrays.rx_pos[view_index, rx_indices_np], dtype=torch.float32, device=device
    )
    freq_indices = torch.as_tensor(freq_indices_np, dtype=torch.long, device=device)
    measured = measured_view(
        arrays, view_index, tx_indices_np, rx_indices_np, freq_indices_np, device
    )

    view = model.view_field(
        tx_pos,
        rx_pos,
        opacity_scale=args.opacity_scale,
        opacity_key=args.opacity_key,
        opacity_normalize=args.opacity_normalize,
        use_lambertian=args.lambertian,
        use_occlusion=args.occlusion,
        occlusion_steps=args.occlusion_steps or None,
        occlusion_step_frac=args.occlusion_step_frac,
        query_chunk=args.query_chunk,
        occlusion_point_chunk=args.occlusion_point_chunk,
    )
    predicted_raw = range_forward_operator(
        frequencies,
        kvector,
        rx_pos,
        tx_pos,
        view["points"],
        view["weights"],
        phase_sign=args.phase_sign,
        freq_indices=freq_indices,
        oversample=args.range_oversample,
        kernel_width=args.range_kernel_width,
        pair_chunk=args.pair_chunk,
        point_chunk=args.point_chunk,
        compute_dtype=torch.float64 if args.compute_dtype == "fp64" else torch.float32,
        range_model=args.range_model,
    )
    gain.maybe_init_scale(predicted_raw, measured.to(predicted_raw.dtype))
    predicted = gain(predicted_raw)
    diff = predicted - measured.to(predicted.dtype)
    data_loss = diff.abs().square().mean()

    regularizers = model.regularizers(view)
    total_loss = data_loss
    if include_regularizers:
        total_loss = (
            total_loss
            + args.sparse_weight * regularizers["sparse"]
            + args.density_tv_weight * regularizers["density_tv"]
            + args.scatter_tv_weight * regularizers["scatter_tv"]
            + args.phase_tv_weight * regularizers["phase_tv"]
        )
    terms = {"data": data_loss, **regularizers}
    return total_loss, terms, metric_record(predicted, measured)


@torch.no_grad()
def evaluate(
    model: SHSASField,
    gain: GlobalComplexGain,
    arrays: RadarFieldsArrays,
    frequencies: torch.Tensor,
    kvector: torch.Tensor,
    view_indices: Sequence[int],
    args,
    device: torch.device,
) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    gain.eval()
    rng = np.random.default_rng(args.seed + 104729)
    tx_indices = select_axis_indices(
        rng, arrays.num_tx, args.eval_tx, random=False
    )
    rx_indices = select_axis_indices(
        rng, arrays.num_rx, args.eval_rx, random=False
    )
    freq_indices = select_frequency_indices(
        rng, arrays.num_freq, args.eval_freq_wanted, random=False
    )
    selected_views = [int(v) for v in view_indices]
    if args.eval_max_views > 0:
        selected_views = selected_views[: args.eval_max_views]

    records = []
    for number, view_index in enumerate(selected_views, start=1):
        if STOP_REQUESTED:
            break
        _loss, _terms, metrics = view_objective(
            model,
            gain,
            arrays,
            frequencies,
            kvector,
            view_index,
            tx_indices,
            rx_indices,
            freq_indices,
            args,
            device,
            include_regularizers=False,
        )
        records.append(metrics)
        if number % 25 == 0:
            print(f"  validation {number}/{len(selected_views)} views", flush=True)
    if was_training:
        model.train()
        gain.train()
    metrics = combine_metrics(records)
    metrics["complete"] = float(len(records) == len(selected_views))
    metrics["views"] = float(len(records))
    return metrics


def atomic_torch_save(payload: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    print(f"Checkpoint saved atomically to {path}", flush=True)


def atomic_json(payload: Dict[str, object], path: Path) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def save_history(history: Sequence[Dict[str, float]], path: Path) -> None:
    if not history:
        return
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(temporary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    os.replace(temporary, path)


def checkpoint_payload(
    model: SHSASField,
    gain: GlobalComplexGain,
    optimizer,
    step: int,
    best_val: float,
    rng: np.random.Generator,
    history: Sequence[Dict[str, float]],
    args,
) -> Dict[str, object]:
    return {
        "sh_sas_contract_version": CONTRACT_VERSION,
        "sealed_npz_protocol_contract": getattr(args, "sealed_npz_protocol_contract", None),
        "step": int(step),
        "epoch": int(step),
        "loss": float(best_val),
        "best_val_rel_mse": float(best_val),
        "scene_repr": "hash_sh",
        "extent": float(args.extent),
        "granularity": int(args.granularity),
        "sh_degree": int(args.sh_degree),
        "geometry_truth_used_for_training": False,
        "paper_architecture": model.paper_architecture,
        "paper_terms": {
            "dc_density": True,
            "dc_normals": True,
            "lambertian": bool(args.lambertian),
            "tx_rx_transmittance": bool(args.occlusion),
            "opacity_key": args.opacity_key,
            "opacity_normalize_radar_adaptation": bool(args.opacity_normalize),
        },
        # Existing geometry tooling consumes these dense coefficient tensors.
        "model_state_dict": model.geometry_compatibility_state(args.query_chunk),
        "sh_sas_state_dict": model.state_dict(),
        "gain_state_dict": gain.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng_state_json": json.dumps(rng.bit_generator.state),
        "history": list(history),
        "args": vars(args),
    }


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz-path", required=True)
    parser.add_argument("--npz-role-manifest", default=None,
                        help="Optional sealed train/validation manifest (required for RIFT dataset)")
    parser.add_argument("--checkpoint-name", default="b787_sh_sas")
    parser.add_argument("--checkpoint-root", default="training_checkpoints")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--num-train", type=int, default=1800)
    parser.add_argument("--num-val", type=int, default=200)
    parser.add_argument("--num-test", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-from-tail", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--extent", type=float, default=0.15)
    parser.add_argument("--granularity", type=int, default=48,
                        help="quadrature/evaluation lattice; the neural field itself is continuous")
    parser.add_argument("--sh-degree", type=int, default=PAPER_SH_DEGREE)
    parser.add_argument("--hidden-dim", type=int, default=PAPER_MLP_WIDTH)
    parser.add_argument("--hash-levels", type=int, default=PAPER_HASH_LEVELS)
    parser.add_argument("--hash-features", type=int, default=2)
    parser.add_argument("--hash-base-resolution", type=int, default=PAPER_HASH_BASE_RESOLUTION)
    parser.add_argument("--hash-final-resolution", type=int, default=PAPER_HASH_FINAL_RESOLUTION)
    parser.add_argument("--hash-log2-size", type=int, default=19)
    parser.add_argument("--query-chunk", type=int, default=32768)

    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--views-per-step", type=int, default=1)
    parser.add_argument("--num-freq-wanted", type=int, default=100)
    parser.add_argument("--train-tx", type=int, default=0, help="0 = all transmitters")
    parser.add_argument("--train-rx", type=int, default=0, help="0 = all receivers")
    parser.add_argument("--eval-freq-wanted", type=int, default=0, help="0 = all frequencies")
    parser.add_argument("--eval-tx", type=int, default=0, help="0 = all transmitters")
    parser.add_argument("--eval-rx", type=int, default=0, help="0 = all receivers")
    parser.add_argument("--eval-max-views", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=10)

    parser.add_argument("--lr", type=float, default=1.0e-3,
                        help="paper uses fixed Adam learning rate 1e-3")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--sparse-weight", type=float, default=0.0)
    parser.add_argument("--density-tv-weight", type=float, default=0.0)
    parser.add_argument("--scatter-tv-weight", type=float, default=0.0)
    parser.add_argument("--phase-tv-weight", type=float, default=0.0,
                        help="paper disables all priors on simulated data; B787 defaults to zero")

    parser.add_argument("--opacity-scale", type=float, default=0.1)
    parser.add_argument("--opacity-key", choices=("dc", "energy"), default="dc")
    parser.add_argument("--opacity-normalize", action=argparse.BooleanOptionalAction, default=True,
                        help="mean-normalized radar adaptation; --no-opacity-normalize is literal Eq. 4")
    parser.add_argument("--lambertian", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--occlusion", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--occlusion-steps", type=int, default=0)
    parser.add_argument("--occlusion-step-frac", type=float, default=0.5)
    parser.add_argument("--occlusion-point-chunk", type=int, default=16384)

    parser.add_argument("--phase-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    parser.add_argument("--range-model", choices=("sum2", "product", "none"), default="sum2")
    parser.add_argument("--compute-dtype", choices=("fp32", "fp64"), default="fp64")
    parser.add_argument("--range-oversample", type=int, default=2)
    parser.add_argument("--range-kernel-width", type=int, default=20)
    parser.add_argument("--pair-chunk", type=int, default=16)
    parser.add_argument("--point-chunk", type=int, default=16384)
    return parser.parse_args(argv)


def validate_args(args) -> None:
    if args.eval_only and not args.resume:
        raise ValueError("--eval-only requires --resume")
    for name in ("steps", "views_per_step", "eval_every", "checkpoint_every"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.sh_degree != PAPER_SH_DEGREE:
        print(
            f"WARNING: SH-SAS paper baseline fixes L=3; requested L={args.sh_degree} is an ablation.",
            flush=True,
        )
    if args.hidden_dim != PAPER_MLP_WIDTH or args.hash_levels != PAPER_HASH_LEVELS:
        print("WARNING: requested network differs from the paper architecture.", flush=True)
    if args.opacity_scale < 0:
        raise ValueError("--opacity-scale must be non-negative")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    set_seed(args.seed)
    device = torch.device(args.device)

    print("SH-SAS independent implementation: arXiv:2509.11087 / 3DV 2026")
    print("Geometry supervision: disabled; initialization: random neural-field parameters")
    print(f"Using device: {device}")
    args.sealed_npz_protocol_contract = None
    if args.npz_role_manifest:
        from train import _load_sealed_npz_protocol_contract
        from rift.radar_fields_dataset import restrict_radar_fields_response_views
        public, contract = _load_sealed_npz_protocol_contract(
            args.npz_path, args.npz_role_manifest, num_train=args.num_train,
            num_val=args.num_val, num_test=args.num_test)
        args.sealed_npz_protocol_contract = contract
        train_indices = np.asarray(contract["role_ids"]["train"], dtype=np.int64)
        val_indices = np.asarray(contract["role_ids"]["validation"], dtype=np.int64)
        if contract.get('dataset_identity'):
            from rift.radar_fields_dataset import from_collection_arrays
            arrays = from_collection_arrays(public, contract)
        else:
            arrays = restrict_radar_fields_response_views(
                load_radar_fields_npz(args.npz_path, load_response=False),
                np.concatenate((train_indices, val_indices)))
    else:
        arrays = load_radar_fields_npz(args.npz_path)
        train_indices, val_indices, _test_indices = split_view_indices(
            arrays.num_views, args.num_train, args.num_val, args.num_test,
            args.seed, val_from_tail=args.val_from_tail)
    frequencies = torch.as_tensor(
        build_frequency_grid(arrays.metadata), dtype=torch.float32, device=device
    )
    kvector = (2.0 * torch.pi * frequencies) / cc
    print(
        f"Dataset: {arrays.num_views} views, {arrays.num_tx}x{arrays.num_rx} pairs, "
        f"{arrays.num_freq} frequencies; split train={len(train_indices)} val={len(val_indices)}"
    )

    checkpoint_dir = Path(args.checkpoint_root) / args.checkpoint_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(args, device)
    gain = GlobalComplexGain().to(device)
    print(f"Paper architecture: {model.paper_architecture}")
    print(
        f"Rendering: DC density/normals, Lambertian={args.lambertian}, "
        f"occlusion={args.occlusion}, key={args.opacity_key}, zeta={args.opacity_scale:g}, "
        f"opacity_normalize={args.opacity_normalize}"
    )
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(gain.parameters()),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1.0e-15,
    )
    rng = np.random.default_rng(args.seed)
    start_step = 0
    best_val = float("inf")
    history = []

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        from train import _validate_saved_sealed_npz_protocol_contract
        _validate_saved_sealed_npz_protocol_contract(
            checkpoint.get("sealed_npz_protocol_contract"), args.sealed_npz_protocol_contract)
        if checkpoint.get("sh_sas_contract_version") != CONTRACT_VERSION:
            raise ValueError("resume checkpoint has an incompatible SH-SAS contract version")
        if checkpoint.get("geometry_truth_used_for_training", True):
            raise ValueError("resume checkpoint does not certify the sensor-only training contract")
        model.load_state_dict(checkpoint["sh_sas_state_dict"])
        gain.load_state_dict(checkpoint["gain_state_dict"])
        if checkpoint.get("optimizer_state_dict") is not None and not args.eval_only:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = int(checkpoint["step"])
        best_val = float(checkpoint.get("best_val_rel_mse", checkpoint.get("loss", float("inf"))))
        history = list(checkpoint.get("history", []))
        rng.bit_generator.state = json.loads(checkpoint["numpy_rng_state_json"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda" and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        print(f"Resumed SH-SAS from step {start_step}, best val rel-MSE={best_val:.6e}")

    if args.eval_only:
        metrics = evaluate(
            model, gain, arrays, frequencies, kvector, val_indices, args, device
        )
        atomic_json(metrics, checkpoint_dir / "sh_sas_eval.json")
        print(
            f"Validation: rel-MSE={metrics['rel_mse']:.6%} "
            f"L1(Re/Im/Abs)={metrics['l1_real']:.3e}/{metrics['l1_imag']:.3e}/"
            f"{metrics['l1_abs']:.3e}"
        )
        return

    model.train()
    gain.train()
    prior_enabled = any(
        weight > 0
        for weight in (
            args.sparse_weight,
            args.density_tv_weight,
            args.scatter_tv_weight,
            args.phase_tv_weight,
        )
    )
    print(f"Paper Eq. (8) priors enabled: {prior_enabled} (paper disables them on simulated data)")

    for step_zero in range(start_step, args.steps):
        step = step_zero + 1
        started = time.time()
        optimizer.zero_grad(set_to_none=True)
        batch_views = rng.choice(
            train_indices,
            size=args.views_per_step,
            replace=len(train_indices) < args.views_per_step,
        )
        records = []
        term_totals = {key: 0.0 for key in ("data", "sparse", "density_tv", "scatter_tv", "phase_tv")}

        completed_views = 0
        for view_index in batch_views:
            tx_indices = select_axis_indices(rng, arrays.num_tx, args.train_tx, random=True)
            rx_indices = select_axis_indices(rng, arrays.num_rx, args.train_rx, random=True)
            freq_indices = select_frequency_indices(
                rng, arrays.num_freq, args.num_freq_wanted, random=True
            )
            loss, terms, metrics = view_objective(
                model,
                gain,
                arrays,
                frequencies,
                kvector,
                int(view_index),
                tx_indices,
                rx_indices,
                freq_indices,
                args,
                device,
                include_regularizers=True,
            )
            (loss / args.views_per_step).backward()
            completed_views += 1
            records.append(metrics)
            for key in term_totals:
                term_totals[key] += float(terms[key].detach()) / args.views_per_step
            if STOP_REQUESTED:
                break

        parameters = list(model.parameters()) + list(gain.parameters())
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
        optimizer.step()
        train_metrics = combine_metrics(records)
        elapsed = time.time() - started
        print(
            f"Step [{step}/{args.steps}] data={term_totals['data']:.6e} "
            f"rel-MSE={train_metrics['rel_mse']:.4%} "
            f"zeta={args.opacity_scale:.3g} |g|={abs(gain.gain_value()):.3e} "
            f"grad={float(grad_norm):.3e} [{elapsed:.1f}s/step]",
            flush=True,
        )
        if prior_enabled:
            print(
                "  priors: "
                + " ".join(f"{key}={term_totals[key]:.3e}" for key in (
                    "sparse", "density_tv", "scatter_tv", "phase_tv"
                )),
                flush=True,
            )

        validation = None
        should_evaluate = (step % args.eval_every == 0 or step == args.steps) and not STOP_REQUESTED
        if should_evaluate:
            validation = evaluate(
                model, gain, arrays, frequencies, kvector, val_indices, args, device
            )
            if validation["complete"]:
                print(
                    f"Validation [step {step}] rel-MSE={validation['rel_mse']:.6%} "
                    f"L1(Re/Im/Abs)={validation['l1_real']:.3e}/"
                    f"{validation['l1_imag']:.3e}/{validation['l1_abs']:.3e}",
                    flush=True,
                )
                row = {
                    "step": step,
                    "train_rel_mse": train_metrics["rel_mse"],
                    "val_rel_mse": validation["rel_mse"],
                    "val_l1_real": validation["l1_real"],
                    "val_l1_imag": validation["l1_imag"],
                    "val_l1_abs": validation["l1_abs"],
                    "val_mse_real": validation["mse_real"],
                    "val_mse_imag": validation["mse_imag"],
                    "val_mse_abs": validation["mse_abs"],
                    "step_seconds": elapsed,
                }
                history.append(row)
                save_history(history, checkpoint_dir / "sh_sas_history.csv")

        is_best = bool(validation and validation["complete"] and validation["rel_mse"] < best_val)
        if is_best:
            best_val = validation["rel_mse"]
        should_checkpoint = (
            step % args.checkpoint_every == 0 or should_evaluate or STOP_REQUESTED
        )
        payload = None
        if should_checkpoint or is_best:
            payload = checkpoint_payload(
                model, gain, optimizer, step, best_val, rng, history, args
            )
        if is_best:
            atomic_torch_save(payload, checkpoint_dir / "checkpoint_best.pth.tar")
        if should_checkpoint:
            name = "checkpoint_final.pth.tar" if step == args.steps else "checkpoint_latest.pth.tar"
            atomic_torch_save(payload, checkpoint_dir / name)

        if STOP_REQUESTED:
            print(
                f"Stopped cleanly after {completed_views} view(s) and publishing checkpoint_latest.",
                flush=True,
            )
            return

    summary = {
        "status": "complete",
        "steps": int(args.steps),
        "best_val_rel_mse": float(best_val),
        "geometry_truth_used_for_training": False,
        "paper_architecture": model.paper_architecture,
    }
    atomic_json(summary, checkpoint_dir / "run_summary.json")
    print("SH-SAS training complete", flush=True)


if __name__ == "__main__":
    main()
