#!/usr/bin/env python
"""Isolated CVDomes Camry initialization audit and corrected RIFT pilots.

The production PublicRadar source snapshot remains untouched.  This entrypoint
is launched with that snapshot first on ``PYTHONPATH`` and reuses its exact NPZ
loader, grid scene, phase-reference geometry, range operator, and BP routine.

Cells:
  initdiag  no-training comparison of current/stratified BP and stored/fitted gain
  deg0      corrected degree-zero 1/5/15-epoch pilot
  deg3      corrected degree-three 1/5/15-epoch pilot
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = PROJECT_ROOT / "rift" / "public_radar_tuning.py"
_HELPER_SPEC = importlib.util.spec_from_file_location(
    "rift_public_radar_tuning_pure", HELPER_PATH
)
_HELPERS = importlib.util.module_from_spec(_HELPER_SPEC)
sys.modules[_HELPER_SPEC.name] = _HELPERS
_HELPER_SPEC.loader.exec_module(_HELPERS)

ComplexAccumulators = _HELPERS.ComplexAccumulators
accumulation_windows = _HELPERS.accumulation_windows
angular_coverage_hole_deg = _HELPERS.angular_coverage_hole_deg
closed_form_gain = _HELPERS.closed_form_gain
epoch_group_order = _HELPERS.epoch_group_order
score_gain = _HELPERS.score_gain
select_group_stratified_fps = _HELPERS.select_group_stratified_fps
validate_embedded_partition = _HELPERS.validate_embedded_partition

# These imports must resolve from the immutable PublicRadar source root.
import train as public_train
from rift.calibration import GlobalComplexGain
from rift.config import cc
from rift.forward_operator import get_kvector
from rift.npz_dataset import PecSphereNPZDataset, load_npz_arrays
from rift.range_operator import range_forward_operator
from rift.sparse_scene import SHVoxelGridScene


SCHEMA = "rift.public_radar_camry_corrected_v1"
DEFAULT_EXTENT = 2.886751345948129
DEFAULT_GRANULARITY = 64
DEFAULT_SEED = 42
DEFAULT_UPDATES = 1800
DEFAULT_GATES = (1, 5, 15)
EXPECTED_CHECKPOINT_VAL_REL_MSE = 1.003738


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_torch_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def assert_public_source_contract():
    expected = Path(os.environ["RIFT_PUBLIC_SOURCE_ROOT"]).resolve()
    actual_train = Path(public_train.__file__).resolve()
    actual_rift = Path(sys.modules["rift"].__file__).resolve()
    if expected not in actual_train.parents or expected not in actual_rift.parents:
        raise RuntimeError(
            f"frozen PublicRadar import drift: root={expected}, "
            f"train={actual_train}, rift={actual_rift}"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("cell", choices=("initdiag", "deg0", "deg3"))
    parser.add_argument("launch_mode", choices=("fresh", "resume"))
    parser.add_argument("--npz-path", required=True)
    parser.add_argument("--checkpoint-best", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--extent", type=float, default=DEFAULT_EXTENT)
    parser.add_argument("--granularity", type=int, default=DEFAULT_GRANULARITY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--bp-views", type=int, default=100)
    parser.add_argument("--updates-per-epoch", type=int, default=DEFAULT_UPDATES)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--scene-lr", type=float, default=3.0e-5)
    parser.add_argument("--adam-eps", type=float, default=1.0e-15)
    parser.add_argument("--point-chunk", type=int, default=262144)
    parser.add_argument("--pair-chunk", type=int, default=64)
    return parser.parse_args()


def load_dataset_contract(path):
    arrays = load_npz_arrays(path)
    required = (
        "frequencies_hz",
        "view_azimuth_deg",
        "view_elevation_deg",
        "split_group_id",
        "train_indices",
        "validation_indices",
        "test_indices",
    )
    missing = [name for name in required if name not in arrays]
    if missing:
        raise ValueError(f"canonical Camry NPZ is missing {missing}")
    metadata = dict(arrays["meta"])
    if metadata.get("schema") != "rift_coherent_radar_v1":
        raise ValueError(f"unexpected dataset schema {metadata.get('schema')!r}")
    if metadata.get("propagation_model") != "monostatic_far_field_reference":
        raise ValueError(
            f"unexpected Camry propagation model {metadata.get('propagation_model')!r}"
        )
    if float(metadata.get("phase_sign")) != -1.0:
        raise ValueError("Camry canonical phase sign must be -1")
    if tuple(float(v) for v in metadata.get("scene_center_m", ())) != (0.0, 0.0, 0.0):
        raise ValueError("Camry scene center contract changed")
    if arrays["response"].shape[1:4] != (1, 1, 1):
        raise ValueError(f"Camry response geometry changed: {arrays['response'].shape}")
    n_view = int(arrays["response"].shape[0])
    expected_shapes = {
        "viewpoint_positions": (n_view, 3),
        "tx_pos": (n_view, 1, 3),
        "rx_pos": (n_view, 1, 3),
    }
    for name, expected_shape in expected_shapes.items():
        if tuple(arrays[name].shape) != expected_shape:
            raise ValueError(f"Camry {name} shape changed: {arrays[name].shape}")
        if not np.isfinite(np.asarray(arrays[name])).all():
            raise ValueError(f"Camry {name} contains a non-finite value")
    response = np.asarray(arrays["response"])
    if not np.isfinite(response.real).all() or not np.isfinite(response.imag).all():
        raise ValueError("Camry response contains a non-finite value")
    frequencies = np.asarray(arrays["frequencies_hz"], dtype=np.float64)
    if frequencies.shape != (512,):
        raise ValueError(f"expected all 512 Camry frequencies, got {frequencies.shape}")

    partition = validate_embedded_partition(
        arrays["response"].shape[0],
        arrays["train_indices"],
        arrays["validation_indices"],
        arrays["test_indices"],
        arrays["split_group_id"],
    )
    if partition["train"] != 20736 or partition["validation"] != 2304:
        raise ValueError(f"Camry embedded split changed: {partition}")
    return arrays, metadata, partition


def select_initialization_views(arrays, bp_views):
    train_indices = np.asarray(arrays["train_indices"], dtype=np.int64)
    first = train_indices[:bp_views].copy()
    stratified = select_group_stratified_fps(
        train_indices,
        arrays["viewpoint_positions"],
        arrays["split_group_id"],
        arrays["view_elevation_deg"],
        count=bp_views,
    )
    elevations = np.asarray(arrays["view_elevation_deg"], dtype=np.float64)
    levels, counts = np.unique(np.round(elevations[stratified], 3), return_counts=True)
    first_hole = angular_coverage_hole_deg(
        np.asarray(arrays["viewpoint_positions"])[train_indices],
        np.asarray(arrays["viewpoint_positions"])[first],
    )
    stratified_hole = angular_coverage_hole_deg(
        np.asarray(arrays["viewpoint_positions"])[train_indices],
        np.asarray(arrays["viewpoint_positions"])[stratified],
    )
    selection = {
        "first_indices": first.tolist(),
        "stratified_indices": stratified.tolist(),
        "first_unique_groups": int(
            np.unique(np.asarray(arrays["split_group_id"])[first]).size
        ),
        "stratified_unique_groups": int(
            np.unique(np.asarray(arrays["split_group_id"])[stratified]).size
        ),
        "stratified_elevation_counts": {
            f"{float(level):.3f}": int(count) for level, count in zip(levels, counts)
        },
        "first_coverage_hole_deg": first_hole,
        "stratified_coverage_hole_deg": stratified_hole,
        "first_azimuth_range_deg": [
            float(np.min(np.asarray(arrays["view_azimuth_deg"])[first])),
            float(np.max(np.asarray(arrays["view_azimuth_deg"])[first])),
        ],
        "stratified_azimuth_range_deg": [
            float(np.min(np.asarray(arrays["view_azimuth_deg"])[stratified])),
            float(np.max(np.asarray(arrays["view_azimuth_deg"])[stratified])),
        ],
    }
    if selection["stratified_unique_groups"] != bp_views:
        raise ValueError("stratified BP selection repeated an acquisition group")
    if sorted(selection["stratified_elevation_counts"].values()) != [25, 25, 25, 25]:
        raise ValueError(
            f"stratified BP elevation quotas changed: "
            f"{selection['stratified_elevation_counts']}"
        )
    if stratified_hole >= 10.0:
        raise ValueError(f"stratified BP coverage hole is {stratified_hole:.3f} degrees")
    return first, stratified, selection


def operator_kwargs(metadata, point_chunk, pair_chunk):
    return {
        "range_model": "none",
        "propagation_model": str(metadata["propagation_model"]),
        "reference_range_m": float(metadata["reference_range_m"]),
        "scene_center_m": tuple(float(v) for v in metadata["scene_center_m"]),
        "point_chunk": int(point_chunk),
        "pair_chunk": int(pair_chunk),
    }


def make_scene(degree, granularity, extent, device):
    return SHVoxelGridScene(
        int(granularity),
        float(extent),
        device,
        max_degree=int(degree),
        init_degree=int(degree),
        init_scale=0.0,
    ).to(device)


def backproject(model, arrays, indices, device, extent, op_kwargs, seed):
    public_train.set_seed(seed)
    dataset = PecSphereNPZDataset(arrays, np.asarray(indices, dtype=np.int64))
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    public_train.backprojection_init(
        model,
        loader,
        device,
        num_freq_selected=512,
        arr_dist=0.0,
        spacing=0.0,
        num_rx=1,
        num_tx=1,
        max_viewpoints=len(dataset),
        phase_sign=-1.0,
        forward_operator_name="range",
        compute_dtype=torch.float64,
        data_format="npz",
        op_kwargs=op_kwargs,
    )
    y00 = 0.5 / np.sqrt(np.pi)
    bp_weight_max = float(
        torch.sqrt(model.w_re[..., 0] ** 2 + model.w_im[..., 0] ** 2).max()
        * y00
    )
    return dataset, bp_weight_max


class Renderer:
    def __init__(self, arrays, metadata, device, op_kwargs):
        self.device = device
        self.op_kwargs = dict(op_kwargs)
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
            raise ValueError("corrected Camry lane is sealed to 1x1 geometry")

    def raw_prediction_and_measurement(self, model, item):
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
        positions, weights = model.active_scatterers(dtheta_batch, dphi_batch)
        predicted = range_forward_operator(
            self.freqs,
            self.kvector,
            rx_pos,
            tx_pos,
            positions,
            weights,
            phase_sign=-1.0,
            freq_indices=self.freq_indices,
            compute_dtype=torch.float64,
            **self.op_kwargs,
        )
        return predicted, measured.to(predicted.dtype)


def accumulate_model(model, dataset, renderer, elevations=None):
    cross = 0.0 + 0.0j
    predicted_power = 0.0
    measured_power = 0.0
    sample_count = 0
    per_elevation = {}
    model.eval()
    with torch.no_grad():
        for local_index in range(len(dataset)):
            predicted, measured = renderer.raw_prediction_and_measurement(
                model, dataset[local_index]
            )
            a = complex((predicted.conj() * measured).sum().item())
            b = float((predicted.real.square() + predicted.imag.square()).sum().item())
            c = float((measured.real.square() + measured.imag.square()).sum().item())
            n = int(measured.numel())
            cross += a
            predicted_power += b
            measured_power += c
            sample_count += n
            if elevations is not None:
                key = f"{float(elevations[local_index]):.3f}"
                row = per_elevation.setdefault(
                    key, {"cross": 0.0 + 0.0j, "predicted": 0.0, "measured": 0.0, "n": 0}
                )
                row["cross"] += a
                row["predicted"] += b
                row["measured"] += c
                row["n"] += n
    total = ComplexAccumulators(
        cross=cross,
        predicted_power=predicted_power,
        measured_power=measured_power,
        sample_count=sample_count,
    ).validate()
    by_elevation = {
        key: ComplexAccumulators(
            cross=value["cross"],
            predicted_power=value["predicted"],
            measured_power=value["measured"],
            sample_count=value["n"],
        ).validate()
        for key, value in per_elevation.items()
    }
    return total, by_elevation


def score_with_elevations(total, per_elevation, gain):
    result = score_gain(total, gain)
    result["per_elevation"] = {
        elevation: score_gain(accumulators, gain)
        for elevation, accumulators in sorted(per_elevation.items())
    }
    return result


def one_view_gain(model, dataset, renderer, device):
    gain = GlobalComplexGain().to(device)
    with torch.no_grad():
        predicted, measured = renderer.raw_prediction_and_measurement(model, dataset[0])
        gain.maybe_init_scale(predicted, measured)
    return gain.gain_value()


def infer_degree(model_state):
    n_basis = int(model_state["w_re"].shape[-1])
    degree = int(round(math.sqrt(n_basis) - 1))
    if (degree + 1) ** 2 != n_basis:
        raise ValueError(f"checkpoint has non-SH coefficient count {n_basis}")
    return degree


def load_existing_checkpoint(path, granularity, extent, device):
    checkpoint = torch.load(
        path, map_location="cpu", mmap=True, weights_only=False
    )
    state = checkpoint["model_state_dict"]
    if checkpoint.get("scene_repr") != "grid_sh":
        raise ValueError("existing Camry checkpoint is not grid_sh")
    if checkpoint.get("range_model") != "none":
        raise ValueError("existing Camry checkpoint does not use range_model=none")
    if int(checkpoint.get("granularity", -1)) != int(granularity):
        raise ValueError("existing Camry checkpoint granularity mismatch")
    if not math.isclose(
        float(checkpoint.get("extent", float("nan"))),
        float(extent),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("existing Camry checkpoint extent mismatch")
    if tuple(state["w_re"].shape[:3]) != (
        int(granularity),
        int(granularity),
        int(granularity),
    ):
        raise ValueError("existing Camry checkpoint scene tensor shape mismatch")
    degree = infer_degree(state)
    model = make_scene(degree, granularity, extent, device)
    model.load_state_dict(state, strict=True)
    gain = GlobalComplexGain().to(device)
    gain.load_state_dict(checkpoint["gain_state_dict"], strict=True)
    return model, gain.gain_value(), int(checkpoint.get("epoch", -1)), degree


def diagnostic(args, arrays, metadata, partition, run_dir, device):
    if (run_dir / "status.json").is_file():
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        if status.get("state") == "complete":
            print(f"Diagnostic already complete: {run_dir / 'status.json'}", flush=True)
            return
    first, stratified, selection = select_initialization_views(arrays, args.bp_views)
    op_kwargs = operator_kwargs(metadata, args.point_chunk, args.pair_chunk)
    renderer = Renderer(arrays, metadata, device, op_kwargs)

    first_model = make_scene(0, args.granularity, args.extent, device)
    first_dataset, first_bp_max = backproject(
        first_model, arrays, first, device, args.extent, op_kwargs, args.seed
    )
    stratified_dataset = PecSphereNPZDataset(arrays, stratified)
    current_gain = one_view_gain(first_model, first_dataset, renderer, device)
    first_fit, _ = accumulate_model(first_model, stratified_dataset, renderer)
    first_multiview_gain = closed_form_gain(first_fit)

    stratified_model = make_scene(0, args.granularity, args.extent, device)
    _, stratified_bp_max = backproject(
        stratified_model, arrays, stratified, device, args.extent, op_kwargs, args.seed
    )
    stratified_fit, _ = accumulate_model(stratified_model, stratified_dataset, renderer)
    stratified_multiview_gain = closed_form_gain(stratified_fit)

    validation_indices = np.asarray(arrays["validation_indices"], dtype=np.int64)
    validation_dataset = PecSphereNPZDataset(arrays, validation_indices)
    validation_elevations = np.asarray(arrays["view_elevation_deg"])[validation_indices]
    first_validation, first_by_elevation = accumulate_model(
        first_model, validation_dataset, renderer, validation_elevations
    )
    stratified_validation, stratified_by_elevation = accumulate_model(
        stratified_model, validation_dataset, renderer, validation_elevations
    )

    checkpoint_model, stored_gain, checkpoint_epoch, checkpoint_degree = load_existing_checkpoint(
        args.checkpoint_best, args.granularity, args.extent, device
    )
    checkpoint_fit, _ = accumulate_model(
        checkpoint_model, stratified_dataset, renderer
    )
    checkpoint_fitted_gain = closed_form_gain(checkpoint_fit)
    checkpoint_validation, checkpoint_by_elevation = accumulate_model(
        checkpoint_model, validation_dataset, renderer, validation_elevations
    )

    zero = ComplexAccumulators(
        cross=0.0j,
        predicted_power=0.0,
        measured_power=first_validation.measured_power,
        sample_count=first_validation.sample_count,
    )
    rows = {
        "zero_predictor": score_with_elevations(zero, {}, 0.0j),
        "first100_bp_internal_alpha": score_with_elevations(
            first_validation, first_by_elevation, 1.0 + 0.0j
        ),
        "first100_bp_current_one_view_gain": score_with_elevations(
            first_validation, first_by_elevation, current_gain
        ),
        "first100_bp_stratified_fit_gain": score_with_elevations(
            first_validation, first_by_elevation, first_multiview_gain
        ),
        "stratified100_bp_internal_alpha": score_with_elevations(
            stratified_validation, stratified_by_elevation, 1.0 + 0.0j
        ),
        "stratified100_bp_stratified_fit_gain": score_with_elevations(
            stratified_validation, stratified_by_elevation, stratified_multiview_gain
        ),
        "checkpoint_best_stored_gain": score_with_elevations(
            checkpoint_validation, checkpoint_by_elevation, stored_gain
        ),
        "checkpoint_best_stratified_fit_gain": score_with_elevations(
            checkpoint_validation, checkpoint_by_elevation, checkpoint_fitted_gain
        ),
    }
    validation_oracle_lower_bounds = {
        "first100_bp": score_gain(
            first_validation, closed_form_gain(first_validation)
        ),
        "stratified100_bp": score_gain(
            stratified_validation, closed_form_gain(stratified_validation)
        ),
        "checkpoint_best": score_gain(
            checkpoint_validation, closed_form_gain(checkpoint_validation)
        ),
    }
    fit_rows = {
        "first100_bp_internal_alpha": score_gain(first_fit, 1.0 + 0.0j),
        "first100_bp_current_one_view_gain": score_gain(first_fit, current_gain),
        "first100_bp_stratified_fit_gain": score_gain(first_fit, first_multiview_gain),
        "stratified100_bp_internal_alpha": score_gain(stratified_fit, 1.0 + 0.0j),
        "stratified100_bp_stratified_fit_gain": score_gain(
            stratified_fit, stratified_multiview_gain
        ),
        "checkpoint_best_stored_gain": score_gain(checkpoint_fit, stored_gain),
        "checkpoint_best_stratified_fit_gain": score_gain(
            checkpoint_fit, checkpoint_fitted_gain
        ),
    }
    for name, fit in fit_rows.items():
        rows[name]["stratified100_fit"] = fit

    engineering = {
        "zero_is_100_percent": abs(rows["zero_predictor"]["relative_mse"] - 1.0) < 1.0e-12,
        "partition_complete_and_group_safe": partition["n_view"] == 23040,
        "stratified_100_unique_groups": selection["stratified_unique_groups"] == 100,
        "stratified_four_equal_elevation_quotas": sorted(
            selection["stratified_elevation_counts"].values()
        ) == [25, 25, 25, 25],
        "stratified_coverage_hole_under_10_deg": selection[
            "stratified_coverage_hole_deg"
        ] < 10.0,
        "ordered_bp_replays_weight_max": abs(first_bp_max / 6.621 - 1.0) < 0.005,
        "ordered_bp_replays_one_view_gain": abs(
            current_gain - complex(1.3714, -0.0068)
        ) / abs(complex(1.3714, -0.0068)) < 0.01,
        "checkpoint_replays_validation": abs(
            rows["checkpoint_best_stored_gain"]["relative_mse"]
            - EXPECTED_CHECKPOINT_VAL_REL_MSE
        ) < 2.0e-4,
        "closed_form_first_bp_improves_fit": rows[
            "first100_bp_stratified_fit_gain"
        ]["stratified100_fit"]["relative_mse"]
        <= rows["first100_bp_current_one_view_gain"]["stratified100_fit"][
            "relative_mse"
        ] + 1.0e-12,
        "closed_form_checkpoint_improves_fit": rows[
            "checkpoint_best_stratified_fit_gain"
        ]["stratified100_fit"]["relative_mse"]
        <= rows["checkpoint_best_stored_gain"]["stratified100_fit"]["relative_mse"]
        + 1.0e-12,
    }
    payload = {
        "schema": SCHEMA,
        "schema_version": 1,
        "cell": "initdiag",
        "dataset": str(Path(args.npz_path).resolve()),
        "source_root": str(Path(os.environ["RIFT_PUBLIC_SOURCE_ROOT"]).resolve()),
        "partition": partition,
        "selection": selection,
        "bp_weight_max": {
            "first100": first_bp_max,
            "stratified100": stratified_bp_max,
        },
        "checkpoint": {
            "path": str(Path(args.checkpoint_best).resolve()),
            "epoch": checkpoint_epoch,
            "allocated_degree": checkpoint_degree,
        },
        "validation_rows": rows,
        # Explicit diagnostic lower bounds only.  These gains are fitted on
        # validation and are never used as reportable model settings or fed to
        # either training arm.
        "validation_oracle_gain_lower_bounds": validation_oracle_lower_bounds,
        "engineering_gates": engineering,
        "scientific_gate": {
            "stratified_bp_fitted_val_below_99_percent": rows[
                "stratified100_bp_stratified_fit_gain"
            ]["relative_mse"] < 0.99,
            "stratified_bp_predicted_power_above_1_percent": rows[
                "stratified100_bp_stratified_fit_gain"
            ]["predicted_to_measured_power"] > 0.01,
            "meaningful_below_95_percent": rows[
                "stratified100_bp_stratified_fit_gain"
            ]["relative_mse"] < 0.95,
        },
        "completed_unix": time.time(),
    }
    atomic_json(run_dir / "diagnostic.json", payload)
    if not all(engineering.values()):
        failed = [name for name, passed in engineering.items() if not passed]
        raise RuntimeError(f"diagnostic engineering gates failed: {failed}")
    atomic_json(
        run_dir / "status.json",
        {
            "schema": SCHEMA,
            "cell": "initdiag",
            "state": "complete",
            "artifact": "diagnostic.json",
            "completed_unix": time.time(),
        },
    )
    print(json.dumps(payload["scientific_gate"], sort_keys=True), flush=True)


def mean_train_target_power(arrays, train_indices, chunk_size=1024):
    response = arrays["response"]
    total_power = 0.0
    total_samples = 0
    for start in range(0, len(train_indices), chunk_size):
        indices = train_indices[start : start + chunk_size]
        cube = np.asarray(response[indices]).mean(axis=3)
        total_power += float(np.square(np.abs(cube).astype(np.float64)).sum())
        total_samples += int(cube.size)
    if total_power <= 0 or total_samples <= 0:
        raise ValueError("training target power is empty")
    return total_power / total_samples, total_power, total_samples


@torch.no_grad()
def balance_scene_and_gain(model, gain):
    squared = model.w_re.square() + model.w_im.square()
    squared = squared.sum(dim=-1)
    rms = float(torch.sqrt(squared.mean()).item())
    if not math.isfinite(rms) or rms <= 0:
        raise ValueError(f"cannot balance scene with RMS {rms}")
    scale = 1.0 / rms
    model.w_re.mul_(scale)
    model.w_im.mul_(scale)
    return complex(gain / scale), rms, scale


def fixed_evaluate(model, gain, dataset, renderer):
    accumulators, _ = accumulate_model(model, dataset, renderer)
    result = score_gain(accumulators, gain)
    result.update(
        {
            "squared_error_sum": float(
                result["relative_mse"] * accumulators.measured_power
            ),
            "target_power_sum": float(accumulators.measured_power),
            "raw_prediction_power_sum": float(accumulators.predicted_power),
            "raw_cross_real": float(accumulators.cross.real),
            "raw_cross_imag": float(accumulators.cross.imag),
        }
    )
    return result


def write_history_csv(path, history):
    path = Path(path)
    fields = (
        "epoch",
        "online_objective_mean",
        "train_relative_mse",
        "validation_relative_mse",
        "gain_magnitude",
        "gain_phase_rad",
        "epoch_seconds",
    )
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in history:
            writer.writerow(
                {
                    "epoch": row["epoch"],
                    "online_objective_mean": row["online_objective_mean"],
                    "train_relative_mse": (
                        row.get("train_fixed", {}).get("relative_mse", "")
                    ),
                    "validation_relative_mse": (
                        row.get("validation_fixed", {}).get("relative_mse", "")
                    ),
                    "gain_magnitude": abs(complex(*row["gain"])),
                    "gain_phase_rad": np.angle(complex(*row["gain"])),
                    "epoch_seconds": row["epoch_seconds"],
                }
            )
    os.replace(temporary, path)


def checkpoint_payload(args, degree, epoch, model, optimizer, gain, history, contract):
    return {
        "schema": SCHEMA,
        "schema_version": 1,
        "cell": args.cell,
        "degree": int(degree),
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "gain_real": float(gain.real),
        "gain_imag": float(gain.imag),
        "gain_frozen": True,
        "history": history,
        "contract": contract,
    }


def run_training(args, arrays, metadata, partition, run_dir, device):
    degree = 0 if args.cell == "deg0" else 3
    status_path = run_dir / "status.json"
    if status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("state") == "complete":
            print(f"Training already complete: {status_path}", flush=True)
            return

    train_indices = np.asarray(arrays["train_indices"], dtype=np.int64)
    validation_indices = np.asarray(arrays["validation_indices"], dtype=np.int64)
    _, stratified, selection = select_initialization_views(arrays, args.bp_views)
    train_groups = np.asarray(arrays["split_group_id"], dtype=np.int64)[train_indices]
    target_mean_power, target_power_sum, target_sample_count = mean_train_target_power(
        arrays, train_indices
    )
    op_kwargs = operator_kwargs(metadata, args.point_chunk, args.pair_chunk)
    renderer = Renderer(arrays, metadata, device, op_kwargs)
    train_dataset = PecSphereNPZDataset(arrays, train_indices)
    validation_dataset = PecSphereNPZDataset(arrays, validation_indices)

    contract = {
        "schema": SCHEMA,
        "cell": args.cell,
        "degree": degree,
        "dataset": str(Path(args.npz_path).resolve()),
        "source_root": str(Path(os.environ["RIFT_PUBLIC_SOURCE_ROOT"]).resolve()),
        "partition": partition,
        "frequencies": 512,
        "granularity": int(args.granularity),
        "extent": float(args.extent),
        "phase_sign": -1.0,
        "range_model": "none",
        "propagation_model": metadata["propagation_model"],
        "reference_range_m": float(metadata["reference_range_m"]),
        "seed": int(args.seed),
        "bp_views": int(args.bp_views),
        "bp_selection": selection,
        "updates_per_epoch": int(args.updates_per_epoch),
        "gradient_reduction": "mean_within_near_equal_windows",
        "objective": "mean_complex_mse_divided_by_fixed_full_train_target_mean_power",
        "target_mean_power": float(target_mean_power),
        "scene_lr": float(args.scene_lr),
        "adam_eps": float(args.adam_eps),
        "point_chunk": int(args.point_chunk),
        "pair_chunk": int(args.pair_chunk),
        "compute_dtype": "float64",
        "scheduler": "constant",
        "gain": "closed_form_on_stratified100_then_gauge_balanced_and_frozen",
        "l1_weight": 0.0,
        "gates": list(DEFAULT_GATES),
        "planned_epochs": int(args.epochs),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    contract_path = run_dir / "contract.json"
    if contract_path.is_file():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing != contract:
            raise ValueError("existing corrected-pilot contract disagrees with this invocation")
    else:
        if args.launch_mode == "resume":
            raise ValueError("resume requested but contract.json is absent")
        atomic_json(contract_path, contract)

    model = make_scene(degree, args.granularity, args.extent, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.scene_lr, eps=args.adam_eps, weight_decay=0.0
    )
    latest_path = run_dir / "checkpoint_latest.pth.tar"
    if args.launch_mode == "resume":
        if not latest_path.is_file():
            raise ValueError("resume requested but checkpoint_latest is absent")
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        if checkpoint.get("schema") != SCHEMA:
            raise ValueError("resume checkpoint schema mismatch")
        if checkpoint.get("cell") != args.cell or int(checkpoint.get("degree", -1)) != degree:
            raise ValueError("resume checkpoint cell/degree mismatch")
        if checkpoint.get("gain_frozen") is not True:
            raise ValueError("resume checkpoint does not preserve the frozen-gain contract")
        if checkpoint.get("contract") != contract:
            raise ValueError("resume checkpoint contract mismatch")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        gain = complex(checkpoint["gain_real"], checkpoint["gain_imag"])
        start_epoch = int(checkpoint["epoch"])
        history = list(checkpoint.get("history", []))
        if not 0 <= start_epoch <= args.epochs:
            raise ValueError(f"resume epoch {start_epoch} is outside [0, {args.epochs}]")
        if len(history) != start_epoch:
            raise ValueError(
                f"resume history has {len(history)} rows for completed epoch {start_epoch}"
            )
        if [int(row.get("epoch", -1)) for row in history] != list(
            range(1, start_epoch + 1)
        ):
            raise ValueError("resume history epochs are not contiguous from 1")
        print(f"Resumed {args.cell} after epoch {start_epoch}", flush=True)
    else:
        if latest_path.exists():
            raise ValueError("fresh launch refuses existing checkpoint_latest")
        stratified_dataset, bp_weight_max = backproject(
            model, arrays, stratified, device, args.extent, op_kwargs, args.seed
        )
        fit_accumulators, _ = accumulate_model(model, stratified_dataset, renderer)
        raw_gain = closed_form_gain(fit_accumulators)
        gain, raw_scene_rms, balance_scale = balance_scene_and_gain(model, raw_gain)
        start_epoch = 0
        history = []
        atomic_json(
            run_dir / "initialization.json",
            {
                "schema": SCHEMA,
                "cell": args.cell,
                "bp_weight_max": bp_weight_max,
                "raw_multiview_gain": [raw_gain.real, raw_gain.imag],
                "raw_scene_rms": raw_scene_rms,
                "scene_balance_scale": balance_scale,
                "balanced_frozen_gain": [gain.real, gain.imag],
                "fit_metrics_after_balance": score_gain(
                    accumulate_model(model, stratified_dataset, renderer)[0], gain
                ),
            },
        )
        # Make even the first epoch restartable: a TERM before epoch 1 reaches
        # its boundary must not strand a manifest-bound run with no checkpoint.
        atomic_torch_save(
            latest_path,
            checkpoint_payload(
                args,
                degree,
                0,
                model,
                optimizer,
                gain,
                history,
                contract,
            ),
        )

    gain_tensor = torch.tensor(gain, dtype=torch.complex128, device=device)
    gates = set(DEFAULT_GATES)
    for epoch_index in range(start_epoch, args.epochs):
        epoch_number = epoch_index + 1
        epoch_start = time.time()
        order = epoch_group_order(train_groups, args.seed, epoch_index)
        windows = accumulation_windows(order, args.updates_per_epoch)
        expected_sizes = sorted(set(int(window.size) for window in windows))
        if expected_sizes != [11, 12]:
            raise RuntimeError(f"unexpected accumulation window sizes {expected_sizes}")
        model.train()
        online_objective = 0.0
        online_views = 0
        for window in windows:
            optimizer.zero_grad(set_to_none=True)
            window_size = int(window.size)
            for local_index in window:
                predicted, measured = renderer.raw_prediction_and_measurement(
                    model, train_dataset[int(local_index)]
                )
                residual = gain_tensor * predicted - measured
                normalized_loss = (
                    (residual.real.square() + residual.imag.square()).mean()
                    / target_mean_power
                )
                online_objective += float(normalized_loss.detach())
                online_views += 1
                (normalized_loss / window_size).backward()
            optimizer.step()

        row = {
            "epoch": epoch_number,
            "online_objective_mean": online_objective / online_views,
            "gain": [gain.real, gain.imag],
            "optimizer_updates": len(windows),
            "views_consumed": online_views,
            "epoch_seconds": time.time() - epoch_start,
        }
        if epoch_number in gates:
            row["train_fixed"] = fixed_evaluate(
                model, gain, train_dataset, renderer
            )
            row["validation_fixed"] = fixed_evaluate(
                model, gain, validation_dataset, renderer
            )
            row["gate_below_zero_predictor"] = {
                "train": row["train_fixed"]["relative_mse"] < 1.0,
                "validation": row["validation_fixed"]["relative_mse"] < 1.0,
            }
            atomic_torch_save(
                run_dir / f"checkpoint_epoch_{epoch_number:03d}.pth.tar",
                checkpoint_payload(
                    args,
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
        atomic_torch_save(
            latest_path,
            checkpoint_payload(
                args,
                degree,
                epoch_number,
                model,
                optimizer,
                gain,
                history,
                contract,
            ),
        )
        atomic_json(run_dir / "history.json", history)
        write_history_csv(run_dir / "history.csv", history)
        print(
            f"{args.cell} epoch {epoch_number}/{args.epochs}: "
            f"online={row['online_objective_mean']:.6f}, "
            f"seconds={row['epoch_seconds']:.1f}"
            + (
                f", train_fixed={row['train_fixed']['relative_mse']:.4%}, "
                f"val_fixed={row['validation_fixed']['relative_mse']:.4%}"
                if "train_fixed" in row
                else ""
            ),
            flush=True,
        )

    gate_rows = [row for row in history if "validation_fixed" in row]
    if [row["epoch"] for row in gate_rows] != list(DEFAULT_GATES):
        raise RuntimeError("training finished without all 1/5/15 fixed-evaluation gates")
    best = min(gate_rows, key=lambda row: row["validation_fixed"]["relative_mse"])
    final = {
        "schema": SCHEMA,
        "schema_version": 1,
        "cell": args.cell,
        "degree": degree,
        "state": "complete",
        "best_gate_epoch": int(best["epoch"]),
        "best_gate_checkpoint": f"checkpoint_epoch_{int(best['epoch']):03d}.pth.tar",
        "best_checkpoint": f"checkpoint_epoch_{int(best['epoch']):03d}.pth.tar",
        "best_train_relative_mse": float(best["train_fixed"]["relative_mse"]),
        "best_train_relative_l2": float(best["train_fixed"]["relative_l2"]),
        "best_validation_relative_mse": float(
            best["validation_fixed"]["relative_mse"]
        ),
        "best_validation_relative_l2": float(
            best["validation_fixed"]["relative_l2"]
        ),
        "final_train_relative_mse": float(
            gate_rows[-1]["train_fixed"]["relative_mse"]
        ),
        "final_train_relative_l2": float(
            gate_rows[-1]["train_fixed"]["relative_l2"]
        ),
        "final_validation_relative_mse": float(
            gate_rows[-1]["validation_fixed"]["relative_mse"]
        ),
        "final_validation_relative_l2": float(
            gate_rows[-1]["validation_fixed"]["relative_l2"]
        ),
        "gain_real": gain.real,
        "gain_imag": gain.imag,
        "gain_frozen": True,
        "target_power_sum": target_power_sum,
        "target_sample_count": target_sample_count,
        "completed_unix": time.time(),
    }
    atomic_json(run_dir / "result.json", final)
    atomic_json(run_dir / "status.json", final)


def main():
    args = parse_args()
    assert_public_source_contract()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one CUDA device is required")
    if args.epochs != 15:
        raise ValueError("corrected Camry pilots are sealed to 15 epochs")
    if args.updates_per_epoch != 1800:
        raise ValueError("corrected Camry pilots are sealed to 1,800 updates/epoch")
    if args.bp_views != 100:
        raise ValueError("corrected Camry pilots are sealed to BP100")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
    arrays, metadata, partition = load_dataset_contract(args.npz_path)
    run_dir = Path(args.run_root).resolve() / args.cell
    if args.cell == "initdiag":
        diagnostic(args, arrays, metadata, partition, run_dir, device)
    else:
        run_training(args, arrays, metadata, partition, run_dir, device)


if __name__ == "__main__":
    main()
