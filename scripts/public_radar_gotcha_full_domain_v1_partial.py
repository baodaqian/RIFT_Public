#!/usr/bin/env python
"""Fresh RIFT pilots on the complete designated GOTCHA pass-2 plane.

The four v1 cells use BP400 or BP1600 and SH degree 0 or 3 on the frozen
100 m by 100 m, z=0 support.  The entire square must pass an exact, fail-closed
range-window preflight before BP.  Every artifact identity and run directory is
new: this trainer cannot load, reshape, or overwrite the retired partial-domain
dense-v2 checkpoints. It preserves training every 30 optimizer updates and
train/validation evaluation every 250 views so Embers interruptions progress.
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
GOTCHA_DOMAIN = _load_local(
    "rift_gotcha_full_domain_v1_training",
    PROJECT_ROOT / "rift" / "gotcha_domain.py",
)
SH_CACHE = _load_local(
    "rift_public_radar_sh_cache_full_domain_v1",
    PROJECT_ROOT / "rift" / "public_radar_sh_cache.py",
)

from rift.config import cc
from rift.forward_operator import get_kvector
from rift.npz_dataset import PecSphereNPZDataset, load_npz_arrays


SCHEMA = "rift.public_radar.gotcha_full_domain_v1"
SCHEMA_VERSION = 1
GATES = (1, 5, 15)
SUPPORTED_POINT_CHUNKS = (32768, 65536, 131072)
PARTIAL_CHECKPOINT_EVERY_UPDATES = 30
EVALUATION_CHECKPOINT_EVERY_VIEWS = 250
SCENES = {
    "gotcha_p2_full_domain": {
        "dataset_name": "gotcha_pass2_hh",
        "parent_metadata": {
            "dataset": "gotcha",
            "pass_id": "pass2",
            "polarization": "hh",
        },
        "propagation_model": "monostatic_near_field_reference",
        "reference_range_m": 51.182879766717654,
        "extent": 50.0,
        "shape": (1776, 1776, 1),
        "frequencies": 426,
        "views": (33939, 4242, 4243),
        "groups": (288, 36, 36),
    },
}
CELL_RE = re.compile(
    r"^(gotcha_p2_full_domain)_bp(400|1600)_deg(0|3)$"
)


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
    parser.add_argument("--view-batch-size", type=int, default=1)
    parser.add_argument("--sh-basis-cache")
    parser.add_argument("--require-sh-basis-cache", action="store_true")
    return parser.parse_args()


def parse_cell(cell):
    match = CELL_RE.fullmatch(cell)
    if match is None:
        raise ValueError(f"invalid GOTCHA full-domain v1 cell {cell!r}")
    scene_key, bp_views, degree = match.groups()
    return scene_key, int(bp_views), int(degree)


def validate_acceleration_args(args):
    view_batch_size = int(getattr(args, "view_batch_size", 1))
    sh_basis_cache = getattr(args, "sh_basis_cache", None)
    require_sh_basis_cache = bool(
        getattr(args, "require_sh_basis_cache", False)
    )
    if view_batch_size <= 0:
        raise ValueError("view_batch_size must be positive")
    if require_sh_basis_cache and not sh_basis_cache:
        raise ValueError(
            "--require-sh-basis-cache requires --sh-basis-cache"
        )
    return view_batch_size, sh_basis_cache, require_sh_basis_cache


def validate_renderer_chunk_options(args):
    """Accept validated opt-in tile sizes while retaining the legacy defaults."""

    point_chunk = getattr(args, "point_chunk", 32768)
    pair_chunk = getattr(args, "pair_chunk", 1)
    if type(point_chunk) is not int or type(pair_chunk) is not int:
        raise ValueError("point_chunk and pair_chunk must be integers")
    if point_chunk not in SUPPORTED_POINT_CHUNKS:
        raise ValueError(
            "point_chunk must be one of "
            f"{SUPPORTED_POINT_CHUNKS}; legacy default is 32768"
        )
    if pair_chunk != 1:
        raise ValueError("pair_chunk must remain 1")
    return point_chunk, pair_chunk


def load_optional_sh_basis_cache(args, arrays, dataset_name):
    _, sh_basis_cache_path, _ = validate_acceleration_args(args)
    if not sh_basis_cache_path:
        return None
    return SH_CACHE.PublicRadarSHBasisCache(
        sh_basis_cache_path,
        arrays["viewpoint_positions"],
        expected_dataset_name=dataset_name,
    )


def validate_full_domain_support(arrays, metadata):
    """Freeze the real pass-2 frequency provenance and complete planar support."""

    archive_frequencies = np.asarray(arrays["frequencies_hz"])
    if archive_frequencies.dtype != np.dtype(np.float64):
        raise ValueError(
            "GOTCHA frequency archive dtype changed: "
            f"{archive_frequencies.dtype}"
        )
    response = np.asarray(arrays["response"])
    viewpoints = np.asarray(arrays["viewpoint_positions"])
    tx_positions = np.asarray(arrays["tx_pos"])
    rx_positions = np.asarray(arrays["rx_pos"])
    if response.dtype != np.dtype(np.complex64):
        raise ValueError(f"GOTCHA response archive dtype changed: {response.dtype}")
    if archive_frequencies.shape != (426,):
        raise ValueError(
            f"GOTCHA frequency archive shape changed: {archive_frequencies.shape}"
        )
    if (
        float(archive_frequencies[0]) != 9_288_080_384.0
        or float(archive_frequencies[-1]) != 9_910_448_128.0
    ):
        raise ValueError("GOTCHA frequency archive endpoints changed")
    if response.shape != (42_424, 1, 1, 1, 426):
        raise ValueError(f"GOTCHA response archive shape changed: {response.shape}")
    if viewpoints.shape != (42_424, 3):
        raise ValueError(f"GOTCHA viewpoint shape changed: {viewpoints.shape}")
    if tx_positions.shape != (42_424, 1, 3):
        raise ValueError(f"GOTCHA Tx geometry shape changed: {tx_positions.shape}")
    if rx_positions.shape != (42_424, 1, 3):
        raise ValueError(f"GOTCHA Rx geometry shape changed: {rx_positions.shape}")
    tx64 = tx_positions[:, 0, :].astype(np.float64, copy=False)
    rx64 = rx_positions[:, 0, :].astype(np.float64, copy=False)
    viewpoints64 = viewpoints.astype(np.float64, copy=False)
    tx_rx_max_offset_m = float(np.linalg.norm(tx64 - rx64, axis=1).max())
    phase_centers = 0.5 * (tx64 + rx64)
    phase_center_viewpoint_max_offset_m = float(
        np.linalg.norm(phase_centers - viewpoints64, axis=1).max()
    )
    if tx_rx_max_offset_m > 1.0e-6:
        raise ValueError(
            "GOTCHA full-domain v1 requires co-located monostatic Tx/Rx; "
            f"max offset is {tx_rx_max_offset_m:.9g} m"
        )
    if phase_center_viewpoint_max_offset_m > 1.0e-3:
        raise ValueError(
            "GOTCHA phase centers disagree with viewpoint positions; "
            f"max offset is {phase_center_viewpoint_max_offset_m:.9g} m"
        )
    frequencies = archive_frequencies.astype(np.float64, copy=False)
    frequency_grid = GOTCHA_DOMAIN.validate_gotcha_frequency_grid(
        frequencies,
        nominal_spacing_hz=float(metadata["frequency_spacing_hz"]),
        unambiguous_range_m=float(metadata["unambiguous_range_m"]),
    )
    preflight = GOTCHA_DOMAIN.validate_gotcha_planar_square_views(
        phase_centers,
        reference_range_m=float(metadata["reference_range_m"]),
        frequency_step_hz=frequency_grid["conservative_spacing_hz"],
        scene_center_m=metadata["scene_center_m"],
    )
    if preflight["mask_realization"] != (
        "identity_after_exact_planar_extrema_preflight"
    ):
        raise ValueError("GOTCHA full-domain planar mask is not the identity")
    support = GOTCHA_DOMAIN.gotcha_full_domain_metadata(
        reference_range_m=float(metadata["reference_range_m"]),
        frequency_step_hz=frequency_grid["nominal_spacing_hz"],
        scene_center_m=metadata["scene_center_m"],
    )
    support["frequency_grid"] = frequency_grid
    support["archive_provenance"] = {
        "frequencies_dtype": str(archive_frequencies.dtype),
        "frequencies_shape": list(archive_frequencies.shape),
        "first_frequency_hz": float(frequencies[0]),
        "last_frequency_hz": float(frequencies[-1]),
        "source_quantization_model": "float32_exact_round_trip",
        "response_dtype": str(response.dtype),
        "response_shape": list(response.shape),
    }
    support["geometry_provenance"] = {
        "preflight_positions": "monostatic_tx_rx_phase_centers",
        "viewpoint_dtype": str(viewpoints.dtype),
        "tx_dtype": str(tx_positions.dtype),
        "rx_dtype": str(rx_positions.dtype),
        "viewpoint_shape": list(viewpoints.shape),
        "tx_shape": list(tx_positions.shape),
        "rx_shape": list(rx_positions.shape),
        "tx_rx_max_offset_m": tx_rx_max_offset_m,
        "tx_rx_tolerance_m": 1.0e-6,
        "phase_center_viewpoint_max_offset_m": (
            phase_center_viewpoint_max_offset_m
        ),
        "phase_center_viewpoint_tolerance_m": 1.0e-3,
    }
    support["preflight"] = preflight
    return support


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
    support = validate_full_domain_support(arrays, metadata)
    return arrays, metadata, partition, support


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
        ("gotcha_p2_full_domain", 400): (288, 1, 2),
        ("gotcha_p2_full_domain", 1600): (288, 5, 6),
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
        raise AssertionError("GOTCHA full-domain v1 is sealed to a z=0 plane")
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
    def __init__(
        self,
        arrays,
        metadata,
        device,
        point_chunk,
        pair_chunk,
        sh_basis_cache=None,
    ):
        self.device = device
        self.point_chunk = int(point_chunk)
        self.sh_basis_cache = sh_basis_cache
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
            raise ValueError("GOTCHA full-domain v1 is sealed to 1x1 geometry")

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
        if self.sh_basis_cache is not None:
            predicted, measured = self.raw_prediction_and_measurement_batch(
                model, [item]
            )
            return predicted[0], measured[0]
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

    def batch_tensors(self, items):
        """Collate strictly aligned 1x1 views for the batched range path."""
        items = list(items)
        if not items:
            raise ValueError("aligned view batches must not be empty")
        dtheta_values = []
        dphi_values = []
        magnitudes = []
        phases = []
        rx_positions = []
        tx_positions = []
        expected_samples = int(self.freqs.numel())
        for item in items:
            _, dphi, dtheta, magnitude, phase, rx_pos, tx_pos = item
            dtheta = torch.as_tensor(dtheta)
            dphi = torch.as_tensor(dphi)
            rx_pos = torch.as_tensor(rx_pos)
            tx_pos = torch.as_tensor(tx_pos)
            magnitude = torch.as_tensor(magnitude)
            phase = torch.as_tensor(phase)
            if dtheta.numel() != 1 or dphi.numel() != 1:
                raise ValueError("each public-radar view must have one direction")
            if rx_pos.shape != (1, 3) or tx_pos.shape != (1, 3):
                raise ValueError("each aligned view must have one Tx and one Rx position")
            if magnitude.numel() != expected_samples or phase.numel() != expected_samples:
                raise ValueError("view response size disagrees with the frequency grid")
            dtheta_values.append(dtheta.reshape(()))
            dphi_values.append(dphi.reshape(()))
            magnitudes.append(magnitude.reshape(expected_samples))
            phases.append(phase.reshape(expected_samples))
            rx_positions.append(rx_pos[0])
            tx_positions.append(tx_pos[0])
        dtheta_batch = torch.stack(dtheta_values).to(self.device)
        dphi_batch = torch.stack(dphi_values).to(self.device)
        magnitude_batch = torch.stack(magnitudes).to(self.device)
        phase_batch = torch.stack(phases).to(self.device)
        measured = torch.polar(magnitude_batch, phase_batch).unsqueeze(-1).unsqueeze(-1)
        return (
            dtheta_batch,
            dphi_batch,
            torch.stack(rx_positions).to(self.device),
            torch.stack(tx_positions).to(self.device),
            measured,
        )

    def raw_prediction_and_measurement_batch(self, model, items):
        items = list(items)
        dtheta, dphi, rx_pos, tx_pos, measured = self.batch_tensors(items)
        if self.sh_basis_cache is None:
            scatterer_chunks = lambda: model.scatterer_view_chunks(
                dtheta, dphi, self.point_chunk
            )
        else:
            view_indices = SH_CACHE.item_view_indices(items)
            basis = self.sh_basis_cache.basis_rows(
                view_indices,
                model.max_degree,
                self.device,
                dtype=model.w_re.dtype,
            )
            scatterer_chunks = lambda: model.scatterer_basis_chunks(
                basis, self.point_chunk
            )
        aligned_kwargs = {
            key: value for key, value in self.op_kwargs.items() if key != "pair_chunk"
        }
        predicted = SERIAL.range_forward_operator_aligned_view_chunks(
            self.freqs,
            self.kvector,
            rx_pos,
            tx_pos,
            scatterer_chunks,
            phase_sign=-1.0,
            freq_indices=self.freq_indices,
            **aligned_kwargs,
        ).unsqueeze(-1).unsqueeze(-1)
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
        "schema_version": SCHEMA_VERSION,
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


def validate_checkpoint_identity(
    checkpoint,
    *,
    cell,
    degree,
    contract,
    model,
    context,
):
    """Reject foreign or reshaped state before loading it into the model."""

    if not isinstance(checkpoint, dict):
        raise ValueError(f"{context} checkpoint is not a dictionary")
    if (
        checkpoint.get("schema") != SCHEMA
        or int(checkpoint.get("schema_version", -1)) != SCHEMA_VERSION
        or checkpoint.get("cell") != cell
        or int(checkpoint.get("degree", -1)) != degree
        or checkpoint.get("contract") != contract
    ):
        raise ValueError(f"{context} checkpoint identity/contract mismatch")
    if checkpoint.get("gain_frozen") is not True:
        raise ValueError(f"{context} checkpoint did not preserve frozen gain")
    gain = complex(checkpoint.get("gain_real", math.nan), checkpoint.get("gain_imag", math.nan))
    if not math.isfinite(gain.real) or not math.isfinite(gain.imag):
        raise ValueError(f"{context} checkpoint gain is non-finite")
    saved_state = checkpoint.get("model_state_dict")
    expected_state = model.state_dict()
    if not isinstance(saved_state, dict) or set(saved_state) != set(expected_state):
        raise ValueError(f"{context} checkpoint model keys changed")
    for name, expected in expected_state.items():
        saved = saved_state[name]
        if (
            not isinstance(saved, torch.Tensor)
            or saved.shape != expected.shape
            or saved.dtype != expected.dtype
        ):
            raise ValueError(
                f"{context} checkpoint tensor {name} has shape/dtype "
                f"{getattr(saved, 'shape', None)}/{getattr(saved, 'dtype', None)}, "
                f"expected {expected.shape}/{expected.dtype}"
            )
    if not isinstance(checkpoint.get("optimizer_state_dict"), dict):
        raise ValueError(f"{context} checkpoint optimizer state is absent")
    return gain


def _fixed_metrics(accumulators, gain):
    result = PURE.score_gain(accumulators, gain)
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


def _evaluation_state(
    *,
    cell,
    contract,
    epoch_number,
    split,
    next_view_index,
    cross,
    predicted_power,
    measured_power,
    sample_count,
    elapsed_seconds,
    train_fixed=None,
    train_runtime_seconds=None,
):
    state = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "cell": cell,
        "contract": contract,
        "epoch": int(epoch_number),
        "split": split,
        "next_view_index": int(next_view_index),
        "cross_real": float(complex(cross).real),
        "cross_imag": float(complex(cross).imag),
        "predicted_power": float(predicted_power),
        "measured_power": float(measured_power),
        "sample_count": int(sample_count),
        "elapsed_seconds": float(elapsed_seconds),
    }
    if train_fixed is not None:
        state["train_fixed"] = train_fixed
        state["train_runtime_seconds"] = float(train_runtime_seconds)
    return state


def _validate_evaluation_state(
    state,
    *,
    cell,
    contract,
    epoch_number,
    split,
    dataset_size,
    samples_per_view,
):
    if not isinstance(state, dict):
        raise ValueError("partial evaluation state is not a dictionary")
    if (
        state.get("schema") != SCHEMA
        or int(state.get("schema_version", -1)) != SCHEMA_VERSION
        or state.get("cell") != cell
        or state.get("contract") != contract
        or int(state.get("epoch", -1)) != epoch_number
        or state.get("split") != split
    ):
        raise ValueError("partial evaluation identity/contract mismatch")
    next_view_index = int(state.get("next_view_index", -1))
    sample_count = int(state.get("sample_count", -1))
    if not 0 <= next_view_index <= dataset_size:
        raise ValueError("partial evaluation view index is outside the split")
    if sample_count != next_view_index * samples_per_view:
        raise ValueError("partial evaluation sample count disagrees with its view prefix")
    numeric = (
        float(state.get("cross_real", math.nan)),
        float(state.get("cross_imag", math.nan)),
        float(state.get("predicted_power", math.nan)),
        float(state.get("measured_power", math.nan)),
        float(state.get("elapsed_seconds", math.nan)),
    )
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("partial evaluation contains a non-finite accumulator")
    if numeric[2] < 0.0 or numeric[3] < 0.0 or numeric[4] < 0.0:
        raise ValueError("partial evaluation contains a negative accumulator")
    if next_view_index > 0 and numeric[3] <= 0.0:
        raise ValueError("partial evaluation has zero measured power after consuming views")
    cross_power = numeric[0] * numeric[0] + numeric[1] * numeric[1]
    cauchy_bound = numeric[2] * numeric[3]
    cauchy_tolerance = 1.0e-10 * max(cauchy_bound, 1.0)
    if cross_power > cauchy_bound + cauchy_tolerance:
        raise ValueError("partial evaluation violates the cross-power Cauchy bound")
    return next_view_index


@torch.no_grad()
def resumable_fixed_evaluate(
    model,
    gain,
    dataset,
    renderer,
    *,
    cell,
    contract,
    epoch_number,
    split,
    state_path,
    initial_state=None,
    train_fixed=None,
    train_runtime_seconds=None,
):
    """Evaluate an exact dataset prefix and atomically persist scalar progress."""

    samples_per_view = int(renderer.freqs.numel()) * renderer.num_rx * renderer.num_tx
    if initial_state is None:
        next_view_index = 0
        cross = 0.0 + 0.0j
        predicted_power = 0.0
        measured_power = 0.0
        sample_count = 0
        elapsed_before = 0.0
    else:
        next_view_index = _validate_evaluation_state(
            initial_state,
            cell=cell,
            contract=contract,
            epoch_number=epoch_number,
            split=split,
            dataset_size=len(dataset),
            samples_per_view=samples_per_view,
        )
        cross = complex(
            float(initial_state["cross_real"]),
            float(initial_state["cross_imag"]),
        )
        predicted_power = float(initial_state["predicted_power"])
        measured_power = float(initial_state["measured_power"])
        sample_count = int(initial_state["sample_count"])
        elapsed_before = float(initial_state["elapsed_seconds"])

    model.eval()
    started = time.time()
    for local_index in range(next_view_index, len(dataset)):
        predicted, measured = renderer.raw_prediction_and_measurement(
            model, dataset[local_index]
        )
        cross += complex((predicted.conj() * measured).sum().item())
        predicted_power += float(predicted.abs().square().sum().item())
        measured_power += float(measured.abs().square().sum().item())
        sample_count += int(measured.numel())
        completed = local_index + 1
        if (
            completed % EVALUATION_CHECKPOINT_EVERY_VIEWS == 0
            or completed == len(dataset)
        ):
            V1.atomic_json(
                state_path,
                _evaluation_state(
                    cell=cell,
                    contract=contract,
                    epoch_number=epoch_number,
                    split=split,
                    next_view_index=completed,
                    cross=cross,
                    predicted_power=predicted_power,
                    measured_power=measured_power,
                    sample_count=sample_count,
                    elapsed_seconds=elapsed_before + time.time() - started,
                    train_fixed=train_fixed,
                    train_runtime_seconds=train_runtime_seconds,
                ),
            )
            print(
                f"{cell} durable epoch {epoch_number} {split} evaluation: "
                f"view {completed}/{len(dataset)}",
                flush=True,
            )

    accumulators = PURE.ComplexAccumulators(
        cross=cross,
        predicted_power=predicted_power,
        measured_power=measured_power,
        sample_count=sample_count,
    ).validate()
    return _fixed_metrics(accumulators, gain), elapsed_before + time.time() - started


def resumable_gate_evaluate(
    model,
    gain,
    train_dataset,
    validation_dataset,
    renderer,
    *,
    cell,
    contract,
    epoch_number,
    state_path,
):
    state = None
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("split") not in ("train", "validation"):
            raise ValueError("partial gate evaluation has an invalid split")

    if state is not None and state["split"] == "validation":
        train_fixed = state.get("train_fixed")
        train_seconds = float(state.get("train_runtime_seconds", math.nan))
        samples_per_view = (
            int(renderer.freqs.numel()) * renderer.num_rx * renderer.num_tx
        )
        required_metrics = (
            "relative_mse",
            "relative_l2",
            "coherent_correlation",
        )
        if (
            not isinstance(train_fixed, dict)
            or not math.isfinite(train_seconds)
            or train_seconds <= 0.0
            or int(train_fixed.get("sample_count", -1))
            != len(train_dataset) * samples_per_view
            or not all(
                math.isfinite(float(train_fixed.get(key, math.nan)))
                for key in required_metrics
            )
        ):
            raise ValueError("validation resume is missing its completed train evaluation")
    else:
        train_fixed, train_seconds = resumable_fixed_evaluate(
            model,
            gain,
            train_dataset,
            renderer,
            cell=cell,
            contract=contract,
            epoch_number=epoch_number,
            split="train",
            state_path=state_path,
            initial_state=state,
        )
        state = _evaluation_state(
            cell=cell,
            contract=contract,
            epoch_number=epoch_number,
            split="validation",
            next_view_index=0,
            cross=0.0 + 0.0j,
            predicted_power=0.0,
            measured_power=0.0,
            sample_count=0,
            elapsed_seconds=0.0,
            train_fixed=train_fixed,
            train_runtime_seconds=train_seconds,
        )
        V1.atomic_json(state_path, state)

    validation_fixed, validation_seconds = resumable_fixed_evaluate(
        model,
        gain,
        validation_dataset,
        renderer,
        cell=cell,
        contract=contract,
        epoch_number=epoch_number,
        split="validation",
        state_path=state_path,
        initial_state=state,
        train_fixed=train_fixed,
        train_runtime_seconds=train_seconds,
    )
    return train_fixed, validation_fixed, train_seconds, validation_seconds


def prepare_run_directory(run_dir, launch_mode, contract):
    """Create or validate one fresh-identity cell directory without overwrites."""

    run_dir = Path(run_dir)
    existing_entries = list(run_dir.iterdir()) if run_dir.is_dir() else []
    if launch_mode == "fresh" and existing_entries:
        raise ValueError("fresh launch refuses a nonempty full-domain run directory")
    run_dir.mkdir(parents=True, exist_ok=True)
    contract_path = run_dir / "contract.json"
    if contract_path.is_file():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing != contract:
            raise ValueError("existing full-domain contract disagrees with this invocation")
    else:
        if launch_mode == "resume" and existing_entries:
            raise ValueError(
                "resume refuses a nonempty run directory without its exact contract"
            )
        if launch_mode == "resume":
            print(
                "Clean interruption occurred before contract creation; "
                "restarting deterministic initialization.",
                flush=True,
            )
        V1.atomic_json(contract_path, contract)
    return run_dir


def run(
    args,
    scene_key,
    bp_views,
    degree,
    arrays,
    metadata,
    partition,
    support,
    device,
    sh_basis_cache=None,
):
    view_batch_size, _, require_sh_basis_cache = validate_acceleration_args(args)
    if require_sh_basis_cache and sh_basis_cache is None:
        raise ValueError("required SH basis cache was not loaded")
    train_indices = np.asarray(arrays["train_indices"], dtype=np.int64)
    validation_indices = np.asarray(arrays["validation_indices"], dtype=np.int64)
    selected, selection = select_bp_views(arrays, scene_key, bp_views)
    train_groups = np.asarray(arrays["split_group_id"], dtype=np.int64)[train_indices]
    target_mean_power, target_power_sum, target_sample_count = V1.mean_train_target_power(
        arrays, train_indices
    )
    train_dataset = PecSphereNPZDataset(arrays, train_indices)
    validation_dataset = PecSphereNPZDataset(arrays, validation_indices)
    if sh_basis_cache is not None:
        viewpoint_positions = arrays["viewpoint_positions"]
        train_dataset = SH_CACHE.IndexedPublicRadarDataset(
            train_dataset, train_indices, viewpoint_positions
        )
        validation_dataset = SH_CACHE.IndexedPublicRadarDataset(
            validation_dataset, validation_indices, viewpoint_positions
        )
    bp_dataset = PecSphereNPZDataset(arrays, selected)
    renderer = SerializedRenderer(
        arrays,
        metadata,
        device,
        args.point_chunk,
        args.pair_chunk,
        sh_basis_cache=sh_basis_cache,
    )

    config = SCENES[scene_key]
    nx, ny, nz = config["shape"]
    contract = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
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
        "partial_checkpoint_every_updates": PARTIAL_CHECKPOINT_EVERY_UPDATES,
        "evaluation_checkpoint_every_views": EVALUATION_CHECKPOINT_EVERY_VIEWS,
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
        "test_policy": "sealed_not_evaluated_during_recipe_screening",
        "phase_sign": -1.0,
        "range_model": "none",
        "propagation_model": metadata["propagation_model"],
        "reference_range_m": float(metadata["reference_range_m"]),
        "scene_support": support,
        "planar_mask_realization": support["preflight"]["mask_realization"],
        "measurement_support_policy": (
            "canonical_raw_phase_history_retained; exterior returns are nuisance"
        ),
        "checkpoint_source_policy": (
            "fresh full-domain state only; retired partial-domain state forbidden"
        ),
    }
    if view_batch_size != 1 or sh_basis_cache is not None:
        contract.update(
            {
                "view_batch_size": view_batch_size,
                "view_batch_operator": "strictly_aligned_independent_views",
                "view_batch_geometry": (
                    "one_canonical_tx_rx_pair_per_view_without_cross_view_pairs"
                ),
                "view_batch_loss_reduction": (
                    "sum_per_view_normalized_losses_divided_by_original_window_size"
                ),
                "directional_sh_basis_source": (
                    "immutable_degree6_cache"
                    if sh_basis_cache is not None
                    else "live_per_batch"
                ),
                "sh_basis_cache_required": require_sh_basis_cache,
                "sh_basis_cache": (
                    sh_basis_cache.contract()
                    if sh_basis_cache is not None
                    else None
                ),
                "canonical_view_index_wrapper": (
                    "train_validation; test_sealed_not_instantiated"
                    if sh_basis_cache is not None
                    else None
                ),
            }
        )
    run_dir = prepare_run_directory(
        Path(args.run_root).resolve() / args.cell,
        args.launch_mode,
        contract,
    )

    status_path = run_dir / "status.json"
    if status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("state") == "complete":
            if (
                status.get("schema") != SCHEMA
                or int(status.get("schema_version", -1)) != SCHEMA_VERSION
                or status.get("cell") != args.cell
                or status.get("contract") != contract
                or not status.get("engineering_gates")
                or not all(status["engineering_gates"].values())
            ):
                raise ValueError(
                    "existing complete status failed full-domain identity/gates"
                )
            print(f"Full-domain cell already complete: {status_path}", flush=True)
            return

    model = make_scene(scene_key, degree, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.scene_lr, eps=args.adam_eps, weight_decay=0.0
    )
    latest_path = run_dir / "checkpoint_latest.pth.tar"
    if args.launch_mode == "resume" and latest_path.is_file():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        gain = validate_checkpoint_identity(
            checkpoint,
            cell=args.cell,
            degree=degree,
            contract=contract,
            model=model,
            context="resume",
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"])
        history = list(checkpoint.get("history", []))
        partial_epoch = checkpoint.get("partial_epoch")
        if not 0 <= start_epoch <= args.epochs:
            raise ValueError(
                f"full-domain resume epoch {start_epoch} is outside [0, {args.epochs}]"
            )
        if len(history) != start_epoch:
            raise ValueError("full-domain resume history/epoch mismatch")
        if [int(row.get("epoch", -1)) for row in history] != list(
            range(1, start_epoch + 1)
        ):
            raise ValueError("full-domain resume history is not contiguous from epoch 1")
        if partial_epoch is not None:
            if not isinstance(partial_epoch, dict):
                raise ValueError("full-domain partial epoch checkpoint is not a dictionary")
            if int(partial_epoch.get("epoch_index", -1)) != start_epoch:
                raise ValueError("full-domain partial epoch does not follow completed history")
            if not math.isfinite(float(partial_epoch.get("online_objective", math.nan))):
                raise ValueError("full-domain partial epoch objective is non-finite")
            if int(partial_epoch.get("online_views", -1)) < 0:
                raise ValueError("full-domain partial epoch view count is invalid")
            if float(partial_epoch.get("elapsed_seconds", -1.0)) < 0.0:
                raise ValueError("full-domain partial epoch elapsed time is invalid")
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
                "contract": contract,
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
    legacy_scalar_execution = (
        view_batch_size == 1 and sh_basis_cache is None
    )
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
                raise ValueError("full-domain partial update index is outside the epoch")
            expected_partial_views = sum(
                int(window.size) for window in windows[:next_update_index]
            )
            online_views = int(partial_epoch["online_views"])
            if online_views != expected_partial_views:
                raise ValueError("full-domain partial epoch view count disagrees with update index")
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
            if legacy_scalar_execution:
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
            else:
                for block_start in range(
                    0, window_size, view_batch_size
                ):
                    block_stop = min(
                        block_start + view_batch_size, window_size
                    )
                    items = [
                        train_dataset[int(local_index)]
                        for local_index in window[block_start:block_stop]
                    ]
                    predicted, measured = (
                        renderer.raw_prediction_and_measurement_batch(
                            model, items
                        )
                    )
                    residual = gain_tensor * predicted - measured
                    normalized_losses = (
                        residual.abs().square().mean(dim=(1, 2, 3))
                        / target_mean_power
                    )
                    online_objective += float(
                        normalized_losses.detach().sum().item()
                    )
                    online_views += len(items)
                    (normalized_losses.sum() / window_size).backward()
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
            raise RuntimeError("full-domain epoch runtime/view accounting failed")
        if epoch_number in GATES:
            evaluation_path = (
                run_dir / f"evaluation_epoch_{epoch_number:03d}_partial.json"
            )
            (
                row["train_fixed"],
                row["validation_fixed"],
                row["train_evaluation_seconds"],
                row["validation_evaluation_seconds"],
            ) = resumable_gate_evaluate(
                model,
                gain,
                train_dataset,
                validation_dataset,
                renderer,
                cell=args.cell,
                contract=contract,
                epoch_number=epoch_number,
                state_path=evaluation_path,
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
        if epoch_number in GATES:
            evaluation_path.unlink(missing_ok=True)
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
        raise RuntimeError("full-domain run finished without all 1/5/15 gates")
    for row in gate_rows:
        for split in ("train_fixed", "validation_fixed"):
            for key in ("relative_mse", "relative_l2", "coherent_correlation"):
                if not math.isfinite(float(row[split][key])):
                    raise RuntimeError(
                        f"full-domain epoch {row['epoch']} {split} {key} is non-finite"
                    )
    best = min(gate_rows, key=lambda row: row["validation_fixed"]["relative_mse"])
    best_epoch = int(best["epoch"])
    best_path = run_dir / f"checkpoint_epoch_{best_epoch:03d}.pth.tar"
    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    best_gain = validate_checkpoint_identity(
        best_checkpoint,
        cell=args.cell,
        degree=degree,
        contract=contract,
        model=model,
        context="validation-selected",
    )
    if best_gain != gain:
        raise ValueError("validation-selected checkpoint gain changed across gates")
    model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
    engineering = {
        "all_gate_runtimes_positive": all(row["epoch_seconds"] > 0 for row in gate_rows),
        "all_gate_evaluation_runtimes_positive": all(
            row["train_evaluation_seconds"] > 0
            and row["validation_evaluation_seconds"] > 0
            for row in gate_rows
        ),
        "all_gate_train_samples_exact": all(
            int(row["train_fixed"]["sample_count"])
            == len(train_dataset) * int(config["frequencies"])
            for row in gate_rows
        ),
        "all_gate_validation_samples_exact": all(
            int(row["validation_fixed"]["sample_count"])
            == len(validation_dataset) * int(config["frequencies"])
            for row in gate_rows
        ),
        "all_primary_metrics_finite": all(
            math.isfinite(float(value))
            for metrics in (best["train_fixed"], best["validation_fixed"])
            for key, value in metrics.items()
            if key in ("relative_mse", "relative_l2", "coherent_correlation")
        ),
        "full_domain_preflight_identity": support["preflight"][
            "mask_realization"
        ]
        == "identity_after_exact_planar_extrema_preflight",
        "full_domain_margins_positive": support["preflight"][
            "minimum_lower_window_margin_m"
        ]
        > 0.0
        and support["preflight"]["minimum_upper_window_margin_m"] > 0.0,
    }
    if not all(engineering.values()):
        failed = [name for name, passed in engineering.items() if not passed]
        raise RuntimeError(f"full-domain engineering gates failed: {failed}")
    result = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "cell": args.cell,
        "state": "complete",
        "scene": scene_key,
        "degree": int(degree),
        "bp_views": int(bp_views),
        "best_gate_epoch": best_epoch,
        "best_checkpoint": best_path.name,
        "best_train": best["train_fixed"],
        "best_validation": best["validation_fixed"],
        "test_policy": "sealed_not_evaluated_during_recipe_screening",
        "test_evaluated": False,
        "gain_real": gain.real,
        "gain_imag": gain.imag,
        "gain_frozen": True,
        "target_power_sum": target_power_sum,
        "target_sample_count": target_sample_count,
        "engineering_gates": engineering,
        "contract": contract,
        "completed_unix": time.time(),
    }
    V1.atomic_json(run_dir / "result.json", result)
    V1.atomic_json(status_path, result)


def main():
    args = parse_args()
    validate_acceleration_args(args)
    scene_key, bp_views, degree = parse_cell(args.cell)
    V1.assert_public_source_contract()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one CUDA device is required")
    if args.seed != 42 or args.epochs != 15 or args.updates_per_epoch != 1800:
        raise ValueError("full-domain v1 is sealed to seed42, 15 epochs, and 1,800 updates/epoch")
    validate_renderer_chunk_options(args)
    if not math.isclose(args.scene_lr, 3.0e-5) or not math.isclose(args.adam_eps, 1.0e-15):
        raise ValueError("full-domain v1 optimizer contract changed")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    arrays, metadata, partition, support = load_dataset_contract(
        args.npz_path, scene_key
    )
    sh_basis_cache = load_optional_sh_basis_cache(
        args, arrays, SCENES[scene_key]["dataset_name"]
    )
    run(
        args,
        scene_key,
        bp_views,
        degree,
        arrays,
        metadata,
        partition,
        support,
        device,
        sh_basis_cache=sh_basis_cache,
    )


if __name__ == "__main__":
    main()
