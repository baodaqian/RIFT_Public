#!/usr/bin/env python
"""RIFT interpolation-only pilots with fixed dense planar sampling.

This is a new experiment identity.  It consumes the materialized v2 split,
uses exactly BP400 or BP1600 training views, and trains a fixed z=0 planar SH
grid with serialized coordinate/weight generation.  It never loads or writes
an existing PublicRadar baseline checkpoint.  Version 2 adds execution-only
partial-epoch checkpoints without changing the optimizer, order, or contract.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_local(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load the completed v1 pilot module only for its audited dataset/evaluation,
# atomic-write, and metric helpers.  Its sealed run_training/main paths are not
# invoked and its artifact identity is untouched.
V1 = _load_local(
    "rift_public_radar_camry_tuning_v1_helpers",
    PROJECT_ROOT / "scripts" / "public_radar_camry_tuning.py",
)
PURE = _load_local(
    "rift_public_radar_tuning_dense_v2",
    PROJECT_ROOT / "rift" / "public_radar_tuning.py",
)
PLANAR = _load_local(
    "rift_fixed_planar_scene_dense_v2",
    PROJECT_ROOT / "rift" / "fixed_planar_scene.py",
)
SERIAL = _load_local(
    "rift_serialized_range_operator_dense_v2",
    PROJECT_ROOT / "rift" / "serialized_range_operator.py",
)

from rift.config import cc
from rift.forward_operator import get_kvector
from rift.npz_dataset import PecSphereNPZDataset, load_npz_arrays


SCHEMA = "rift.public_radar_interpolation_dense_v2"
GATES = (1, 5, 15)
PARTIAL_CHECKPOINT_EVERY_UPDATES = 180
SCENES = {
    "camry": {
        "dataset_name": "cvdomes_camry",
        "parent_metadata": {"dataset": "cvdomes", "polarization": "hh"},
        "propagation_model": "monostatic_far_field_reference",
        "reference_range_m": 7.152066310594172,
        "extent": 2.886751345948129,
        "shape": (960, 960, 1),
        "frequencies": 512,
        "views": (18432, 2304, 2304),
        "groups": (576, 72, 72),
    },
    "gotcha_p2": {
        "dataset_name": "gotcha_pass2_hh",
        "parent_metadata": {
            "dataset": "gotcha",
            "pass_id": "pass2",
            "polarization": "hh",
        },
        "propagation_model": "monostatic_near_field_reference",
        "reference_range_m": 51.182879766717654,
        "extent": 28.86751345948129,
        "shape": (1024, 1024, 1),
        "frequencies": 426,
        "views": (33939, 4242, 4243),
        "groups": (288, 36, 36),
    },
}
CELL_RE = re.compile(r"^(camry|gotcha_p2)_bp(400|1600)_deg(0|3)$")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("cell")
    parser.add_argument("launch_mode", choices=("fresh", "resume"))
    parser.add_argument("--npz-path", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--updates-per-epoch", type=int, default=1800)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--scene-lr", type=float, default=3.0e-5)
    parser.add_argument("--adam-eps", type=float, default=1.0e-15)
    parser.add_argument("--point-chunk", type=int, default=32768)
    parser.add_argument("--pair-chunk", type=int, default=1)
    return parser.parse_args()


def parse_cell(cell):
    match = CELL_RE.fullmatch(cell)
    if match is None:
        raise ValueError(f"invalid dense-v2 cell {cell!r}")
    scene_key, bp_views, degree = match.groups()
    return scene_key, int(bp_views), int(degree)


def load_dataset_contract(path, scene_key):
    arrays = load_npz_arrays(path)
    config = SCENES[scene_key]
    metadata = dict(arrays["meta"])
    if metadata.get("schema") != "rift_coherent_radar_v1":
        raise ValueError(f"unexpected dataset schema {metadata.get('schema')!r}")
    if metadata.get("split_schema") != "rift.public_radar_interleaved_split_v2":
        raise ValueError("dataset is not the materialized interleaved v2 split")
    if metadata.get("split_scene") != config["dataset_name"]:
        raise ValueError("interleaved dataset scene identity mismatch")
    for key, expected_value in config["parent_metadata"].items():
        if metadata.get(key) != expected_value:
            raise ValueError(
                f"interleaved parent metadata {key} changed: {metadata.get(key)!r}"
            )
    parent_path = Path(str(metadata.get("split_parent_npz", "")))
    expected_tail = (config["dataset_name"], f"{config['dataset_name']}.npz")
    if tuple(parent_path.parts[-2:]) != expected_tail:
        raise ValueError(
            f"interleaved parent path identity changed: {parent_path}"
        )
    if metadata.get("split_interpolation_only") is not True:
        raise ValueError("interleaved split did not assert interpolation-only use")
    if int(metadata.get("split_seed", -1)) != 42:
        raise ValueError("interleaved split seed changed")
    if float(metadata.get("phase_sign")) != -1.0:
        raise ValueError("public-radar phase sign must be -1")
    if metadata.get("propagation_model") != config["propagation_model"]:
        raise ValueError("public-radar propagation model changed")
    if not math.isclose(
        float(metadata.get("reference_range_m", float("nan"))),
        float(config["reference_range_m"]),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("public-radar reference range changed")
    if tuple(float(v) for v in metadata.get("scene_center_m", ())) != (0.0, 0.0, 0.0):
        raise ValueError("public-radar scene center changed")
    if arrays["response"].shape[1:4] != (1, 1, 1):
        raise ValueError(f"expected 1x1 public-radar response, got {arrays['response'].shape}")
    if int(arrays["response"].shape[-1]) != config["frequencies"]:
        raise ValueError("public-radar frequency count changed")
    n_view = int(arrays["response"].shape[0])
    partition = PURE.validate_embedded_partition(
        n_view,
        arrays["train_indices"],
        arrays["validation_indices"],
        arrays["test_indices"],
        arrays["split_group_id"],
    )
    expected_roles, expected_partition = PURE.interleaved_group_split(
        arrays["split_group_id"], arrays["view_azimuth_deg"], seed=42
    )
    for role in ("train", "validation", "test"):
        actual_indices = np.asarray(arrays[f"{role}_indices"], dtype=np.int64)
        if not np.array_equal(actual_indices, expected_roles[role]):
            raise ValueError(
                f"embedded {role} indices do not replay the sealed angular interleave"
            )
    if metadata.get("split_strategy") != expected_partition["strategy"]:
        raise ValueError("interleaved split strategy metadata disagrees with replay")
    actual_views = tuple(
        int(partition[role]) for role in ("train", "validation", "test")
    )
    actual_groups = tuple(
        int(partition["groups_by_role"][role])
        for role in ("train", "validation", "test")
    )
    if actual_views != config["views"] or actual_groups != config["groups"]:
        raise ValueError(
            f"{scene_key} v2 split changed: views={actual_views}, groups={actual_groups}"
        )
    for name in ("viewpoint_positions", "tx_pos", "rx_pos"):
        if not np.isfinite(np.asarray(arrays[name])).all():
            raise ValueError(f"{name} contains a non-finite value")
    response = np.asarray(arrays["response"])
    if not np.isfinite(response.real).all() or not np.isfinite(response.imag).all():
        raise ValueError("response contains a non-finite value")
    return arrays, metadata, partition


def select_bp_views(arrays, scene_key, bp_views):
    train = np.asarray(arrays["train_indices"], dtype=np.int64)
    selected = PURE.select_balanced_direction_fps(
        train,
        arrays["viewpoint_positions"],
        arrays["split_group_id"],
        count=bp_views,
    )
    train_set = set(int(value) for value in train)
    if any(int(value) not in train_set for value in selected):
        raise AssertionError("BP selection leaked outside the train role")
    groups = np.asarray(arrays["split_group_id"], dtype=np.int64)[selected]
    _, counts = np.unique(groups, return_counts=True)
    expected = {
        ("camry", 400): (400, 1, 1),
        ("camry", 1600): (576, 2, 3),
        ("gotcha_p2", 400): (288, 1, 2),
        ("gotcha_p2", 1600): (288, 5, 6),
    }[(scene_key, bp_views)]
    actual = (int(np.unique(groups).size), int(counts.min()), int(counts.max()))
    if actual != expected:
        raise ValueError(f"{scene_key} BP{bp_views} group rounds changed: {actual}")
    selection = {
        "count": int(bp_views),
        "indices": selected.tolist(),
        "unique_groups": actual[0],
        "min_views_per_selected_group": actual[1],
        "max_views_per_selected_group": actual[2],
        "coverage_hole_deg": PURE.angular_coverage_hole_deg(
            np.asarray(arrays["viewpoint_positions"])[train],
            np.asarray(arrays["viewpoint_positions"])[selected],
        ),
        "azimuth_min_deg": float(np.min(np.asarray(arrays["view_azimuth_deg"])[selected])),
        "azimuth_max_deg": float(np.max(np.asarray(arrays["view_azimuth_deg"])[selected])),
        "elevation_min_deg": float(np.min(np.asarray(arrays["view_elevation_deg"])[selected])),
        "elevation_max_deg": float(np.max(np.asarray(arrays["view_elevation_deg"])[selected])),
    }
    return selected, selection


def make_scene(scene_key, degree, device):
    config = SCENES[scene_key]
    nx, ny, nz = config["shape"]
    if nz != 1:
        raise AssertionError("dense v2 is sealed to a z=0 plane")
    return PLANAR.FixedPlanarSHScene(
        nx,
        ny,
        config["extent"],
        degree,
        device,
        init_scale=0.0,
    ).to(device)


def operator_kwargs(metadata, pair_chunk):
    return {
        "range_model": "none",
        "propagation_model": str(metadata["propagation_model"]),
        "reference_range_m": float(metadata["reference_range_m"]),
        "scene_center_m": tuple(float(v) for v in metadata["scene_center_m"]),
        "pair_chunk": int(pair_chunk),
        "compute_dtype": torch.float64,
    }


class SerializedRenderer:
    def __init__(self, arrays, metadata, device, point_chunk, pair_chunk):
        self.device = device
        self.point_chunk = int(point_chunk)
        self.op_kwargs = operator_kwargs(metadata, pair_chunk)
        self.freqs = torch.as_tensor(
            np.asarray(arrays["frequencies_hz"], dtype=np.float64),
            dtype=torch.float64,
            device=device,
        )
        self.kvector = get_kvector(self.freqs, cc)
        self.freq_indices = torch.arange(self.freqs.numel(), device=device)
        self.num_rx = int(arrays["rx_pos"].shape[1])
        self.num_tx = int(arrays["tx_pos"].shape[1])
        if self.num_rx != 1 or self.num_tx != 1:
            raise ValueError("dense v2 is sealed to 1x1 public-radar geometry")

    def tensors(self, item):
        _, dphi, dtheta, magnitude, phase, rx_pos, tx_pos = item
        dphi_batch = dphi.unsqueeze(0).to(self.device)
        dtheta_batch = dtheta.unsqueeze(0).to(self.device)
        rx_pos = rx_pos.to(self.device)
        tx_pos = tx_pos.to(self.device)
        magnitude_cube = (
            magnitude.to(self.device)
            .view(-1, self.num_tx, self.num_rx)
            .permute(2, 1, 0)
        )
        phase_cube = (
            phase.to(self.device)
            .view(-1, self.num_tx, self.num_rx)
            .permute(2, 1, 0)
        )
        measured = torch.polar(magnitude_cube, phase_cube).permute(2, 0, 1)
        return dtheta_batch, dphi_batch, rx_pos, tx_pos, measured

    def raw_prediction_and_measurement(self, model, item):
        dtheta, dphi, rx_pos, tx_pos, measured = self.tensors(item)
        predicted = SERIAL.range_forward_operator_chunks(
            self.freqs,
            self.kvector,
            rx_pos,
            tx_pos,
            lambda: model.scatterer_chunks(dtheta, dphi, self.point_chunk),
            phase_sign=-1.0,
            freq_indices=self.freq_indices,
            **self.op_kwargs,
        )
        return predicted, measured.to(predicted.dtype)


@torch.no_grad()
def backproject(model, dataset, renderer):
    backprojection = torch.zeros(
        model.n_points, dtype=torch.complex128, device=renderer.device
    )
    for local_index in range(len(dataset)):
        _, _, rx_pos, tx_pos, measured = renderer.tensors(dataset[local_index])
        backprojection += SERIAL.range_adjoint_operator_chunks(
            renderer.freqs,
            renderer.kvector,
            rx_pos,
            tx_pos,
            lambda: model.position_chunks(renderer.point_chunk),
            measured,
            phase_sign=-1.0,
            freq_indices=renderer.freq_indices,
            **renderer.op_kwargs,
        )

    cross = 0.0 + 0.0j
    predicted_power = 0.0
    measured_power = 0.0
    sample_count = 0
    for local_index in range(len(dataset)):
        _, _, rx_pos, tx_pos, measured = renderer.tensors(dataset[local_index])

        def chunks():
            for start in range(0, model.n_points, renderer.point_chunk):
                stop = min(start + renderer.point_chunk, model.n_points)
                yield model.position_chunk(start, stop), backprojection[start:stop]

        predicted = SERIAL.range_forward_operator_chunks(
            renderer.freqs,
            renderer.kvector,
            rx_pos,
            tx_pos,
            chunks,
            phase_sign=-1.0,
            freq_indices=renderer.freq_indices,
            **renderer.op_kwargs,
        )
        cross += complex((predicted.conj() * measured).sum().item())
        predicted_power += float(predicted.abs().square().sum().item())
        measured_power += float(measured.abs().square().sum().item())
        sample_count += int(measured.numel())
    if predicted_power <= 0 or measured_power <= 0:
        raise ValueError("BP fit has empty predicted/measured power")
    alpha = complex(cross / predicted_power)
    backprojection.mul_(torch.tensor(alpha, dtype=backprojection.dtype, device=renderer.device))
    y00 = 0.5 / np.sqrt(np.pi)
    model.w_re.zero_()
    model.w_im.zero_()
    model.w_re[:, 0].copy_(backprojection.real.to(model.w_re.dtype) / y00)
    model.w_im[:, 0].copy_(backprojection.imag.to(model.w_im.dtype) / y00)
    fit = PURE.score_gain(
        PURE.ComplexAccumulators(
            cross=cross,
            predicted_power=predicted_power,
            measured_power=measured_power,
            sample_count=sample_count,
        ),
        alpha,
    )
    return {
        "internal_alpha": [alpha.real, alpha.imag],
        "fit_metrics": fit,
        "effective_weight_max": float(backprojection.abs().max().item()),
    }


def checkpoint_payload(
    cell,
    degree,
    epoch,
    model,
    optimizer,
    gain,
    history,
    contract,
    partial_epoch=None,
):
    return {
        "schema": SCHEMA,
        "schema_version": 1,
        "cell": cell,
        "degree": int(degree),
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "gain_real": float(gain.real),
        "gain_imag": float(gain.imag),
        "gain_frozen": True,
        "history": history,
        "contract": contract,
        "partial_epoch": partial_epoch,
    }


def run(args, scene_key, bp_views, degree, arrays, metadata, partition, device):
    run_dir = Path(args.run_root).resolve() / args.cell
    status_path = run_dir / "status.json"
    if status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("state") == "complete":
            if (
                status.get("schema") != SCHEMA
                or status.get("cell") != args.cell
                or not status.get("engineering_gates")
                or not all(status["engineering_gates"].values())
            ):
                raise ValueError("existing complete status failed dense-v2 identity/gates")
            print(f"Dense-v2 cell already complete: {status_path}", flush=True)
            return

    train_indices = np.asarray(arrays["train_indices"], dtype=np.int64)
    validation_indices = np.asarray(arrays["validation_indices"], dtype=np.int64)
    test_indices = np.asarray(arrays["test_indices"], dtype=np.int64)
    selected, selection = select_bp_views(arrays, scene_key, bp_views)
    train_groups = np.asarray(arrays["split_group_id"], dtype=np.int64)[train_indices]
    target_mean_power, target_power_sum, target_sample_count = V1.mean_train_target_power(
        arrays, train_indices
    )
    train_dataset = PecSphereNPZDataset(arrays, train_indices)
    validation_dataset = PecSphereNPZDataset(arrays, validation_indices)
    test_dataset = PecSphereNPZDataset(arrays, test_indices)
    bp_dataset = PecSphereNPZDataset(arrays, selected)
    renderer = SerializedRenderer(
        arrays, metadata, device, args.point_chunk, args.pair_chunk
    )

    config = SCENES[scene_key]
    nx, ny, nz = config["shape"]
    contract = {
        "schema": SCHEMA,
        "cell": args.cell,
        "scene": scene_key,
        "dataset_name": config["dataset_name"],
        "dataset": str(Path(args.npz_path).resolve()),
        "source_root": str(Path(os.environ["RIFT_PUBLIC_SOURCE_ROOT"]).resolve()),
        "partition": partition,
        "split_schema": metadata["split_schema"],
        "split_strategy": metadata["split_strategy"],
        "split_interpolation_only": True,
        "degree": int(degree),
        "scene_repr": "fixed_planar_sh",
        "shape": [nx, ny, nz],
        "points": int(nx * ny * nz),
        "extent_xy_m": float(config["extent"]),
        "z_m": 0.0,
        "pitch_x_m": float(2.0 * config["extent"] / nx),
        "pitch_y_m": float(2.0 * config["extent"] / ny),
        "adaptive_densification": False,
        "learn_positions": False,
        "pruning": False,
        "angular_growth": False,
        "bp_views": int(bp_views),
        "bp_selection": selection,
        "seed": int(args.seed),
        "updates_per_epoch": int(args.updates_per_epoch),
        "planned_epochs": int(args.epochs),
        "gates": list(GATES),
        "scene_lr": float(args.scene_lr),
        "adam_eps": float(args.adam_eps),
        "point_chunk": int(args.point_chunk),
        "pair_chunk": int(args.pair_chunk),
        "coordinate_materialization": "lazy_per_point_chunk",
        "effective_weight_materialization": "lazy_per_point_chunk",
        "compute_dtype": "float64",
        "objective": "mean_complex_mse_divided_by_fixed_full_train_target_mean_power",
        "gradient_reduction": "mean_within_near_equal_windows",
        "target_mean_power": float(target_mean_power),
        "scheduler": "constant",
        "gain": "bp_internal_closed_form_then_gauge_balanced_and_frozen",
        "test_policy": "once_after_validation_selected_gate",
        "phase_sign": -1.0,
        "range_model": "none",
        "propagation_model": metadata["propagation_model"],
        "reference_range_m": float(metadata["reference_range_m"]),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    contract_path = run_dir / "contract.json"
    if contract_path.is_file():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing != contract:
            raise ValueError("existing dense-v2 contract disagrees with this invocation")
    else:
        if args.launch_mode == "resume":
            print(
                "Clean interruption occurred before contract creation; "
                "restarting deterministic initialization.",
                flush=True,
            )
        V1.atomic_json(contract_path, contract)

    model = make_scene(scene_key, degree, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.scene_lr, eps=args.adam_eps, weight_decay=0.0
    )
    latest_path = run_dir / "checkpoint_latest.pth.tar"
    if args.launch_mode == "resume" and latest_path.is_file():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        if checkpoint.get("schema") != SCHEMA or checkpoint.get("cell") != args.cell:
            raise ValueError("dense-v2 resume checkpoint identity mismatch")
        if int(checkpoint.get("degree", -1)) != degree:
            raise ValueError("dense-v2 resume checkpoint degree mismatch")
        if checkpoint.get("gain_frozen") is not True:
            raise ValueError("dense-v2 resume checkpoint did not preserve frozen gain")
        if checkpoint.get("contract") != contract:
            raise ValueError("dense-v2 resume checkpoint contract mismatch")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        gain = complex(checkpoint["gain_real"], checkpoint["gain_imag"])
        start_epoch = int(checkpoint["epoch"])
        history = list(checkpoint.get("history", []))
        partial_epoch = checkpoint.get("partial_epoch")
        if not 0 <= start_epoch <= args.epochs:
            raise ValueError(
                f"dense-v2 resume epoch {start_epoch} is outside [0, {args.epochs}]"
            )
        if len(history) != start_epoch:
            raise ValueError("dense-v2 resume history/epoch mismatch")
        if [int(row.get("epoch", -1)) for row in history] != list(
            range(1, start_epoch + 1)
        ):
            raise ValueError("dense-v2 resume history is not contiguous from epoch 1")
        if partial_epoch is not None:
            if not isinstance(partial_epoch, dict):
                raise ValueError("dense-v2 partial epoch checkpoint is not a dictionary")
            if int(partial_epoch.get("epoch_index", -1)) != start_epoch:
                raise ValueError("dense-v2 partial epoch does not follow completed history")
            if not math.isfinite(float(partial_epoch.get("online_objective", math.nan))):
                raise ValueError("dense-v2 partial epoch objective is non-finite")
            if int(partial_epoch.get("online_views", -1)) < 0:
                raise ValueError("dense-v2 partial epoch view count is invalid")
            if float(partial_epoch.get("elapsed_seconds", -1.0)) < 0.0:
                raise ValueError("dense-v2 partial epoch elapsed time is invalid")
            print(
                f"Resumed {args.cell} inside epoch {start_epoch + 1} after "
                f"update {int(partial_epoch.get('next_update_index', -1))}",
                flush=True,
            )
        else:
            print(f"Resumed {args.cell} after epoch {start_epoch}", flush=True)
    else:
        if latest_path.exists():
            raise ValueError("fresh launch refuses existing checkpoint_latest")
        if args.launch_mode == "resume":
            print(
                "Clean interruption occurred during initialization; "
                "restarting deterministic BP because no checkpoint exists.",
                flush=True,
            )
        init_start = time.time()
        initialization = backproject(model, bp_dataset, renderer)
        gain, raw_scene_rms, balance_scale = PLANAR.balance_fixed_scene_and_gain(
            model, 1.0 + 0.0j
        )
        initialization.update(
            {
                "bp_views": int(bp_views),
                "bp_selection": selection,
                "raw_scene_rms": raw_scene_rms,
                "scene_balance_scale": balance_scale,
                "balanced_frozen_gain": [gain.real, gain.imag],
                "runtime_seconds": time.time() - init_start,
            }
        )
        if initialization["runtime_seconds"] <= 0:
            raise RuntimeError("BP initialization reported zero runtime")
        V1.atomic_json(run_dir / "initialization.json", initialization)
        start_epoch = 0
        history = []
        partial_epoch = None
        V1.atomic_torch_save(
            latest_path,
            checkpoint_payload(
                args.cell, degree, 0, model, optimizer, gain, history, contract
            ),
        )

    gain_tensor = torch.tensor(gain, dtype=torch.complex128, device=device)
    for epoch_index in range(start_epoch, args.epochs):
        epoch_number = epoch_index + 1
        epoch_start = time.time()
        order = PURE.epoch_group_order(train_groups, args.seed, epoch_index)
        windows = PURE.accumulation_windows(order, args.updates_per_epoch)
        expected_sizes = sorted(
            set(int(window.size) for window in windows)
        )
        base, remainder = divmod(len(train_dataset), args.updates_per_epoch)
        mathematically_expected = sorted(set((base, base + 1 if remainder else base)))
        if expected_sizes != mathematically_expected:
            raise RuntimeError(
                f"unexpected accumulation window sizes {expected_sizes}"
            )
        model.train()
        if partial_epoch is not None and epoch_index == start_epoch:
            next_update_index = int(partial_epoch["next_update_index"])
            if not 0 <= next_update_index <= len(windows):
                raise ValueError("dense-v2 partial update index is outside the epoch")
            expected_partial_views = sum(
                int(window.size) for window in windows[:next_update_index]
            )
            online_views = int(partial_epoch["online_views"])
            if online_views != expected_partial_views:
                raise ValueError("dense-v2 partial epoch view count disagrees with update index")
            online_objective = float(partial_epoch["online_objective"])
            elapsed_before_resume = float(partial_epoch["elapsed_seconds"])
        else:
            next_update_index = 0
            online_objective = 0.0
            online_views = 0
            elapsed_before_resume = 0.0
        for update_index in range(next_update_index, len(windows)):
            window = windows[update_index]
            optimizer.zero_grad(set_to_none=True)
            window_size = int(window.size)
            for local_index in window:
                predicted, measured = renderer.raw_prediction_and_measurement(
                    model, train_dataset[int(local_index)]
                )
                residual = gain_tensor * predicted - measured
                normalized_loss = (
                    residual.abs().square().mean() / target_mean_power
                )
                online_objective += float(normalized_loss.detach())
                online_views += 1
                (normalized_loss / window_size).backward()
            optimizer.step()
            completed_updates = update_index + 1
            if (
                completed_updates % PARTIAL_CHECKPOINT_EVERY_UPDATES == 0
                or completed_updates == len(windows)
            ):
                partial_epoch = {
                    "epoch_index": int(epoch_index),
                    "next_update_index": int(completed_updates),
                    "online_objective": float(online_objective),
                    "online_views": int(online_views),
                    "elapsed_seconds": float(
                        elapsed_before_resume + time.time() - epoch_start
                    ),
                }
                V1.atomic_torch_save(
                    latest_path,
                    checkpoint_payload(
                        args.cell,
                        degree,
                        epoch_index,
                        model,
                        optimizer,
                        gain,
                        history,
                        contract,
                        partial_epoch=partial_epoch,
                    ),
                )
                print(
                    f"{args.cell} durable epoch {epoch_number} progress: "
                    f"update {completed_updates}/{len(windows)}",
                    flush=True,
                )

        row = {
            "epoch": epoch_number,
            "online_objective_mean": online_objective / online_views,
            "gain": [gain.real, gain.imag],
            "optimizer_updates": len(windows),
            "views_consumed": online_views,
            "epoch_seconds": elapsed_before_resume + time.time() - epoch_start,
        }
        if (
            row["epoch_seconds"] <= 0
            or online_views != len(train_dataset)
            or not math.isfinite(row["online_objective_mean"])
        ):
            raise RuntimeError("dense-v2 epoch runtime/view accounting failed")
        if epoch_number in GATES:
            row["train_fixed"] = V1.fixed_evaluate(
                model, gain, train_dataset, renderer
            )
            row["validation_fixed"] = V1.fixed_evaluate(
                model, gain, validation_dataset, renderer
            )
            row["gate_below_zero_predictor"] = {
                "train": row["train_fixed"]["relative_mse"] < 1.0,
                "validation": row["validation_fixed"]["relative_mse"] < 1.0,
            }
            V1.atomic_torch_save(
                run_dir / f"checkpoint_epoch_{epoch_number:03d}.pth.tar",
                checkpoint_payload(
                    args.cell,
                    degree,
                    epoch_number,
                    model,
                    optimizer,
                    gain,
                    history + [row],
                    contract,
                ),
            )
        history.append(row)
        partial_epoch = None
        V1.atomic_torch_save(
            latest_path,
            checkpoint_payload(
                args.cell,
                degree,
                epoch_number,
                model,
                optimizer,
                gain,
                history,
                contract,
            ),
        )
        V1.atomic_json(run_dir / "history.json", history)
        V1.write_history_csv(run_dir / "history.csv", history)
        print(
            f"{args.cell} epoch {epoch_number}/{args.epochs}: "
            f"online={row['online_objective_mean']:.6f}, "
            f"seconds={row['epoch_seconds']:.1f}"
            + (
                f", train={row['train_fixed']['relative_mse']:.4%}, "
                f"validation={row['validation_fixed']['relative_mse']:.4%}"
                if "train_fixed" in row
                else ""
            ),
            flush=True,
        )

    gate_rows = [row for row in history if "validation_fixed" in row]
    if [int(row["epoch"]) for row in gate_rows] != list(GATES):
        raise RuntimeError("dense-v2 finished without all 1/5/15 gates")
    for row in gate_rows:
        for split in ("train_fixed", "validation_fixed"):
            for key in ("relative_mse", "relative_l2", "coherent_correlation"):
                if not math.isfinite(float(row[split][key])):
                    raise RuntimeError(
                        f"dense-v2 epoch {row['epoch']} {split} {key} is non-finite"
                    )
    best = min(gate_rows, key=lambda row: row["validation_fixed"]["relative_mse"])
    best_epoch = int(best["epoch"])
    best_path = run_dir / f"checkpoint_epoch_{best_epoch:03d}.pth.tar"
    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    if best_checkpoint.get("contract") != contract:
        raise ValueError("validation-selected checkpoint contract mismatch")
    if best_checkpoint.get("gain_frozen") is not True:
        raise ValueError("validation-selected checkpoint did not preserve frozen gain")
    best_gain = complex(
        best_checkpoint["gain_real"], best_checkpoint["gain_imag"]
    )
    if best_gain != gain:
        raise ValueError("validation-selected checkpoint gain changed across gates")
    model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
    test_start = time.time()
    test_metrics = V1.fixed_evaluate(model, best_gain, test_dataset, renderer)
    test_seconds = time.time() - test_start
    expected_samples = len(test_dataset) * int(config["frequencies"])
    engineering = {
        "all_gate_runtimes_positive": all(row["epoch_seconds"] > 0 for row in gate_rows),
        "train_samples_exact": int(best["train_fixed"]["sample_count"])
        == len(train_dataset) * int(config["frequencies"]),
        "validation_samples_exact": int(best["validation_fixed"]["sample_count"])
        == len(validation_dataset) * int(config["frequencies"]),
        "test_samples_exact": int(test_metrics["sample_count"]) == expected_samples,
        "test_runtime_positive": test_seconds > 0,
        "all_primary_metrics_finite": all(
            math.isfinite(float(value))
            for metrics in (best["train_fixed"], best["validation_fixed"], test_metrics)
            for key, value in metrics.items()
            if key in ("relative_mse", "relative_l2", "coherent_correlation")
        ),
    }
    if not all(engineering.values()):
        failed = [name for name, passed in engineering.items() if not passed]
        raise RuntimeError(f"dense-v2 engineering gates failed: {failed}")
    result = {
        "schema": SCHEMA,
        "schema_version": 1,
        "cell": args.cell,
        "state": "complete",
        "scene": scene_key,
        "degree": int(degree),
        "bp_views": int(bp_views),
        "best_gate_epoch": best_epoch,
        "best_checkpoint": best_path.name,
        "best_train": best["train_fixed"],
        "best_validation": best["validation_fixed"],
        "test_at_validation_selected_gate": test_metrics,
        "test_runtime_seconds": test_seconds,
        "gain_real": gain.real,
        "gain_imag": gain.imag,
        "gain_frozen": True,
        "target_power_sum": target_power_sum,
        "target_sample_count": target_sample_count,
        "engineering_gates": engineering,
        "completed_unix": time.time(),
    }
    V1.atomic_json(run_dir / "result.json", result)
    V1.atomic_json(status_path, result)


def main():
    args = parse_args()
    scene_key, bp_views, degree = parse_cell(args.cell)
    V1.assert_public_source_contract()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one CUDA device is required")
    if args.seed != 42 or args.epochs != 15 or args.updates_per_epoch != 1800:
        raise ValueError("dense-v2 is sealed to seed42, 15 epochs, and 1,800 updates/epoch")
    if args.point_chunk != 32768 or args.pair_chunk != 1:
        raise ValueError("dense-v2 is sealed to point_chunk=32768 and pair_chunk=1")
    if not math.isclose(args.scene_lr, 3.0e-5) or not math.isclose(args.adam_eps, 1.0e-15):
        raise ValueError("dense-v2 optimizer contract changed")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    arrays, metadata, partition = load_dataset_contract(args.npz_path, scene_key)
    run(args, scene_key, bp_views, degree, arrays, metadata, partition, device)


if __name__ == "__main__":
    main()
