#!/usr/bin/env python
"""CPU gates for the radar-only Radar Fields port.

When ``--upstream`` points at the pinned official checkout, Stage A imports the
released ``rcs_to_intensity`` function and compares our implementation directly
against it. Remaining stages use synthetic data and do not touch a real npz.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rift.encoding import generate_dynamic_grid  # noqa: E402
from rift.radar_fields import (  # noqa: E402
    OFFICIAL_REFERENCE_COMMIT,
    RadarFieldsModel,
    bistatic_range_cells,
    padded_roi_objective_diagnostics,
    radar_fields_intensity,
    radar_fields_loss,
)
from rift.radar_fields_dataset import (  # noqa: E402
    LIGHT_SPEED,
    dataset_provenance,
    load_or_create_stats,
    load_radar_fields_npz,
    load_radar_fields_sealed_split_manifest,
    normalize_power_db,
    range_bin_size,
    restrict_radar_fields_response_views,
    response_view_to_range_power,
    split_view_indices,
)
import rift.radar_fields_dataset as radar_fields_dataset_module  # noqa: E402
import train_radar_fields as radar_fields_train_entrypoint  # noqa: E402
from train_radar_fields import (  # noqa: E402
    atomic_torch_save,
    checkpoint_provenance,
    checkpoint_payload,
    normalize_cuda_rng_state,
    occupancy_compatibility_state,
    resolve_diagnostic_view,
    role_provenance,
    validate_resume_checkpoint,
)


CHECKS = 0


def check(condition, message):
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(message)
    print(f"  PASS: {message}")


def stage_intensity(upstream):
    print("Stage A: released intensity transform parity")
    rcs = torch.tensor([0.0, 0.1, 1.0, 10.0], dtype=torch.float32)
    ranges = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    released = radar_fields_intensity(rcs, ranges, offset=0.05, scaler=1.3, range_law="released")
    expected = torch.log10(rcs + 0.05) * 1.3
    check(torch.allclose(released, expected), "released approximate path is exact")

    code_r2 = radar_fields_intensity(rcs, ranges, offset=0.05, scaler=1.3, range_law="code_r2")
    expected_r2 = torch.log10(rcs / ranges.square() + 0.05) * 1.3
    check(torch.allclose(code_r2, expected_r2), "released non-approximate R^-2 path is exact")

    paper_r4 = radar_fields_intensity(rcs, ranges, range_law="paper_r4")
    check(torch.allclose(paper_r4, torch.log10(rcs / ranges.pow(4) + 0.05)), "paper Eq. 9 R^-4 is explicit")

    if upstream:
        sys.path.insert(0, os.path.abspath(upstream))
        try:
            official = importlib.import_module("radarfields.radar")
            check(
                torch.equal(released, official.rcs_to_intensity(rcs, ranges, 0.05, 1.3, True)),
                "approximate transform matches the pinned official Python function bit-for-bit",
            )
            check(
                torch.equal(code_r2, official.rcs_to_intensity(rcs, ranges, 0.05, 1.3, False)),
                "non-approximate transform matches the pinned official Python function bit-for-bit",
            )
        finally:
            sys.path.pop(0)
    else:
        print("  SKIP upstream import (pass --upstream external/RadarFields_reference to require it)")


def stage_ifft():
    print("Stage B: coherent frequency response -> range-power target")
    count = 64
    bandwidth = 3.0e9
    center = 10.0e9
    dr = LIGHT_SPEED / (2.0 * bandwidth)
    target_bin = 17
    one_way_range = target_bin * dr
    frequencies = (center - bandwidth / 2.0) + np.arange(count) * (bandwidth / count)
    response = np.exp(-1j * 2.0 * np.pi * frequencies * (2.0 * one_way_range) / LIGHT_SPEED)
    view = response.astype(np.complex64).reshape(1, 1, 1, count)
    power = response_view_to_range_power(view)
    check(int(power.argmax(dim=-1).item()) == target_bin, "IFFT peak lands at the expected one-way range bin")
    normalized = normalize_power_db(power, float(power.max()), dynamic_range_db=60.0)
    check(float(normalized.max()) == 1.0 and float(normalized.min()) >= 0.0, "fixed dB normalization maps to [0,1]")


def stage_projector():
    print("Stage C: bistatic range-cell projection")
    xyz = torch.tensor([[1.25, 0.0, 0.0]], requires_grad=False)
    values = torch.tensor([[2.0, 3.0]], requires_grad=True)
    tx = torch.zeros((1, 3))
    rx = torch.zeros((1, 3))
    cells, mass = bistatic_range_cells(
        values, xyz, tx, rx, torch.tensor([0]), bin_size=1.0, num_bins=4, pair_chunk=1
    )
    check(torch.allclose(mass[0, 1:3], torch.tensor([0.75, 0.25])), "fractional range weights are correct")
    check(torch.allclose(cells[0, 1], values[0]) and torch.allclose(cells[0, 2], values[0]), "single-sample cell average is value preserving")
    cells.sum().backward()
    check(values.grad is not None and torch.isfinite(values.grad).all() and values.grad.abs().sum() > 0, "range cells preserve autograd")


def tiny_model():
    return RadarFieldsModel(
        extent=1.0,
        hidden_dim=16,
        feature_dim=8,
        sh_degree=2,
        batch_norm=False,
        hash_levels=2,
        hash_features=2,
        hash_base_resolution=4,
        hash_final_resolution=8,
        hash_log2_size=6,
    )


def stage_model_and_loss():
    print("Stage D: random field, published losses, and gradients")
    torch.manual_seed(4)
    model = tiny_model()
    xyz = generate_dynamic_grid(3, 1.0, torch.device("cpu"), jitter=False).reshape(-1, 3)
    directions = xyz - torch.tensor([2.0, 0.0, 0.0])[None, :]
    out = model(xyz, directions, mask_progress=0.3)
    check(out["rcs"].shape == (xyz.shape[0],), "per-point viewing directions preserve field shape")
    check(((out["alpha"] > 0) & (out["alpha"] < 1)).all(), "random occupancy is strictly in (0,1)")
    check((out["reflectance"] > 0).all(), "random reflectance is nonnegative")

    tx = torch.tensor([[2.0, 0.0, 0.0]])
    rx = torch.tensor([[2.0, 0.0, 0.0]])
    cells, mass = bistatic_range_cells(
        torch.stack((out["alpha"], out["rcs"]), -1),
        xyz,
        tx,
        rx,
        torch.tensor([0]),
        bin_size=0.5,
        num_bins=8,
        pair_chunk=1,
    )
    valid = mass > 0
    pred_occ = cells[..., 0] * valid
    pred = radar_fields_intensity(cells[..., 1] * valid, torch.arange(8)[None] * 0.5 + 0.25)
    target = torch.linspace(0.0, 1.0, 8)[None]
    target_occ = (target >= 0.25).float() * valid
    loss, terms = radar_fields_loss(pred, target, pred_occ, target_occ)
    loss.backward()
    grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    check(all(torch.isfinite(grad).all() for grad in grads), "all field gradients are finite")
    check(sum(float(grad.abs().sum()) for grad in grads) > 0, "published loss terms drive the random field")
    check(set(terms) == {"fft", "occupancy", "bimodal"}, "only the three published loss families are present")


def stage_padded_roi_diagnostic():
    print("Stage E: padded ROI objective diagnostic")
    raw_rcs = torch.nn.Parameter(torch.tensor(0.5))
    valid = torch.tensor([[True, False]])
    # This mirrors the production masking point: a modeled cell is first
    # rendered, then zeroed when no grid sample supported that range bin.
    pred_rcs = raw_rcs.expand(1, 2) * valid.to(torch.float32)
    pred = radar_fields_intensity(pred_rcs, torch.ones((1, 2)), offset=0.05)
    target = torch.tensor([[0.8, 0.8]])
    diagnostics = padded_roi_objective_diagnostics(
        pred,
        target,
        valid,
        [raw_rcs],
        weight_fft=0.60,
    )
    full_l1 = torch.nn.functional.l1_loss(pred, target)
    check(
        abs(
            diagnostics["fft_l1_full_mean"] - float(full_l1.detach())
        ) < 1.0e-7,
        "padded and valid partitions reconstruct the existing FFT L1 reduction",
    )
    check(
        diagnostics["padded_fft_l1_full_mean"] > 0.0
        and diagnostics["padded_roi_fraction"] == 0.5,
        "padded target cells contribute nonzero existing FFT loss",
    )
    check(
        diagnostics["valid_fft_model_gradient_l2"] > 0.0
        and diagnostics["padded_fft_model_gradient_l2"] == 0.0,
        "the synthetic padded contribution has no model gradient after zero masking",
    )
    check(raw_rcs.grad is None, "diagnostic gradient probes do not populate parameter .grad")
    full_l1.backward()
    check(
        raw_rcs.grad is not None and float(raw_rcs.grad.abs()) > 0.0,
        "diagnostic probes retain the graph for the unchanged training backward pass",
    )


def stage_metadata_and_lazy_reader():
    print("Stage F: metadata decoding and diagnostic-safe lazy response access")
    metadata = {
        "radar_fc_hz": 10.0e9,
        "radar_bandwidth_hz": 3.0e9,
        "num_adc_samples": 4,
    }
    response = np.zeros((3, 1, 1, 1, 4), dtype=np.complex64)
    for view in range(response.shape[0]):
        response[view, 0, 0, 0] = np.complex64(view + 1)
    poses = np.arange(9, dtype=np.float32).reshape(3, 3)
    with tempfile.TemporaryDirectory(prefix="rift_radar_fields_metadata_") as temporary:
        for name, encoded_metadata in (
            ("scalar", np.array(json.dumps(metadata))),
            ("singleton", np.array([json.dumps(metadata)])),
            ("bytes_singleton", np.array([json.dumps(metadata).encode("utf-8")])),
        ):
            path = os.path.join(temporary, f"{name}.npz")
            np.savez(
                path,
                response=response,
                viewpoint_positions=poses,
                tx_pos=poses[:, None, :],
                rx_pos=poses[:, None, :],
                metadata_json=encoded_metadata,
            )
            arrays = load_radar_fields_npz(path, load_response=False)
            check(arrays.metadata == metadata, f"{name} metadata_json encoding decodes to the JSON object")
            check(
                arrays.response is None and not arrays.response_is_materialized,
                f"{name} lazy loader retains no full response payload",
            )
            train, val, test = split_view_indices(3, 1, 1, 1, seed=7)
            selected = resolve_diagnostic_view("train", 0, train, val)
            check(selected == int(train[0]), f"{name} diagnostic selection uses an explicit train role ID")
            try:
                resolve_diagnostic_view("test", 0, train, val)
            except ValueError:
                rejected_test = True
            else:
                rejected_test = False
            check(rejected_test, f"{name} diagnostic helper rejects test-role materialization")
            selected_view = arrays.response_view(selected)
            check(
                np.array_equal(selected_view, response[selected]),
                f"{name} lazy reader streams exactly the selected response view",
            )
            check(
                arrays.response is None and not arrays.response_is_materialized,
                f"{name} streamed view does not retain the full response payload",
            )
            streamed = list(arrays.iter_response_views([2, 0, 2]))
            check(
                [index for index, _view in streamed] == [0, 2]
                and np.array_equal(streamed[0][1], response[0])
                and np.array_equal(streamed[1][1], response[2]),
                f"{name} normalization reader streams requested views once without full retention",
            )
            provenance = dataset_provenance(arrays)
            check(
                provenance["response_payload_materialized"] is False
                and provenance["response_shape"] == [3, 1, 1, 1, 4],
                f"{name} provenance records lazy payload state and response contract",
            )
            roles = role_provenance(train, val, test, test_payload_materialized=False)
            check(
                roles["role_ids"]["test"] == [int(test[0])]
                and roles["test_payload_materialized"] is False,
                f"{name} role provenance keeps test identity distinct from payload access",
            )

        eager = load_radar_fields_npz(os.path.join(temporary, "scalar.npz"))
        check(
            eager.response is not None and np.array_equal(eager.response, response),
            "legacy eager loader remains backward-compatible",
        )


def stage_geometry_compatibility():
    print("Stage G: existing B787 geometry-benchmark compatibility")
    model = tiny_model()
    xyz = generate_dynamic_grid(4, 1.0, torch.device("cpu"), jitter=False).reshape(-1, 3)
    state = occupancy_compatibility_state(model, xyz, granularity=4, query_chunk=17)
    check(state["w_re"].shape == (4, 4, 4, 1), "checkpoint occupancy has grid_sh-compatible shape")
    check(torch.count_nonzero(state["w_im"]) == 0, "compatibility field contains no fabricated phase")
    energy = (state["w_re"].square() + state["w_im"].square()).sum(-1)
    check(torch.isfinite(energy).all() and energy.max() > 0, "geometry energy is finite and nonempty")


def stage_checkpoint_roundtrip():
    print("Stage H: atomic checkpoint and radar-only resume contract")
    torch.manual_seed(9)
    model = tiny_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    xyz = generate_dynamic_grid(3, 1.0, torch.device("cpu"), jitter=False).reshape(-1, 3)
    rng = np.random.default_rng(9)
    args = SimpleNamespace(
        extent=1.0,
        granularity=3,
        query_chunk=11,
        hidden_dim=16,
        feature_dim=8,
        sh_degree=2,
        sigmoid_tightness=1.0,
        no_batch_norm=True,
        hash_levels=2,
        hash_features=2,
        hash_base_resolution=4,
        hash_final_resolution=8,
        hash_log2_size=6,
        dynamic_range_db=60.0,
        range_margin=0.05,
        range_law="released",
        intensity_offset=0.05,
        intensity_scaler=1.0,
        num_train=1,
        num_val=1,
        num_test=1,
        seed=9,
        val_from_tail=True,
        steps=10,
        lr=1.0e-3,
        view_batch=1,
        train_pairs=1,
        val_pairs=1,
        eval_every=2,
        eval_max_views=1,
        weight_fft=0.60,
        weight_occ=0.36,
        weight_bimodal=0.03,
        occupancy_threshold=0.05,
        pair_chunk=1,
    )
    dataset_info = {
        "dataset_path": "/synthetic/radar_fields.npz",
        "dataset_file_size_bytes": 1234,
        "response_shape": [3, 1, 1, 1, 4],
        "response_dtype": "complex64",
        "viewpoint_count": 3,
        "tx_count": 1,
        "rx_count": 1,
        "frequency_count": 4,
    }
    split_info = role_provenance(
        [0], [1], [2], test_payload_materialized=True
    )
    stats = {"peak_power": 2.0, "dynamic_range_db": 60.0}
    support_grid_info = {
        "support_xyz_m": [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]],
        "readout_granularity": 3,
    }
    signal_info = {
        "target_domain": "normalized_dB_range_power",
        "normalization": {"peak_power": 2.0, "dynamic_range_db": 60.0},
    }
    payload = checkpoint_payload(
        model,
        optimizer,
        scheduler,
        step=7,
        best_val=0.25,
        xyz=xyz,
        stats=stats,
        rng=rng,
        history=[{"step": 7}],
        args=args,
        dataset_info=dataset_info,
        split_info=split_info,
        support_grid_info=support_grid_info,
        signal_info=signal_info,
    )
    with tempfile.TemporaryDirectory(prefix="rift_radar_fields_") as temporary:
        path = os.path.join(temporary, "checkpoint_latest.pth.tar")
        atomic_torch_save(payload, path=Path(path))
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        provenance = checkpoint_provenance(path, loaded)
    check(loaded["step"] == 7, "atomic checkpoint preserves the optimizer step")
    check(
        loaded["radar_fields_reference_commit"] == OFFICIAL_REFERENCE_COMMIT,
        "checkpoint pins the official reference commit",
    )
    check(loaded["auxiliary_geometry_used"] is False, "checkpoint certifies that no auxiliary geometry was used")
    check(
        loaded["optimizer_state_dict"] is not None and loaded["scheduler_state_dict"] is not None,
        "checkpoint carries optimizer and scheduler resume state",
    )
    check(
        loaded["dataset_provenance"] == dataset_info
        and loaded["split_provenance"] == split_info
        and loaded["support_grid_provenance"] == support_grid_info
        and loaded["signal_provenance"] == signal_info,
        "checkpoint persists dataset, role, support/grid, and signal-normalization provenance",
    )
    restored = tiny_model()
    restored.load_state_dict(loaded["radar_fields_state_dict"])
    check(
        all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), restored.state_dict().values())),
        "checkpoint restores the Radar Fields parameters exactly",
    )
    noncontiguous_rng = torch.arange(16, dtype=torch.uint8)[::2]
    normalized_rng = normalize_cuda_rng_state(
        (noncontiguous_rng,), expected_device_count=1, require_present=True
    )
    check(
        len(normalized_rng) == 1
        and normalized_rng[0].device.type == "cpu"
        and normalized_rng[0].is_contiguous()
        and torch.equal(normalized_rng[0], noncontiguous_rng),
        "CUDA RNG normalization returns contiguous CPU byte tensors without changing state bytes",
    )
    compatibility = validate_resume_checkpoint(
        loaded, args, stats, dataset_info, split_info, resume_device=torch.device("cpu")
    )
    check(
        compatibility["power_stats_verified"]
        and compatibility["dataset_provenance_verified"]
        and compatibility["dataset_identity_verified"],
        "checkpoint compatibility validates persisted model args, power stats, and dataset identity",
    )
    check(
        compatibility["strict_resume_contract"]
        and compatibility["split_provenance_verified"]
        and compatibility["continuation_state_verified"],
        "new checkpoint validates exact split IDs and continuation state before restore",
    )
    check(
        compatibility["cuda_rng_state_verified"] is False,
        "strict CPU resume does not require or consume CUDA RNG state",
    )
    if torch.cuda.is_available():
        saved_cuda_rng = loaded.get("cuda_rng_state")
        check(
            isinstance(saved_cuda_rng, (list, tuple))
            and len(saved_cuda_rng) == torch.cuda.device_count(),
            "CUDA checkpoint captures one RNG state per visible device",
        )
        mapped_cuda_rng = [state.to(device="cuda") for state in saved_cuda_rng]
        normalized_mapped_cuda_rng = normalize_cuda_rng_state(
            mapped_cuda_rng,
            expected_device_count=torch.cuda.device_count(),
            require_present=True,
        )
        check(
            all(state.device.type == "cpu" for state in normalized_mapped_cuda_rng)
            and all(
                torch.equal(expected, restored_state)
                for expected, restored_state in zip(saved_cuda_rng, normalized_mapped_cuda_rng)
            ),
            "CUDA-mapped checkpoint RNG tensors are restored to identical CPU byte tensors",
        )
        original_cuda_rng = torch.cuda.get_rng_state_all()
        try:
            torch.cuda.set_rng_state_all(normalized_mapped_cuda_rng)
            roundtrip_cuda_rng = torch.cuda.get_rng_state_all()
        finally:
            torch.cuda.set_rng_state_all(original_cuda_rng)
        check(
            all(
                torch.equal(expected, restored_state)
                for expected, restored_state in zip(saved_cuda_rng, roundtrip_cuda_rng)
            ),
            "CUDA accepts the normalized CPU RNG tensors and restores every visible-device state",
        )
        cuda_compatibility = validate_resume_checkpoint(
            loaded,
            args,
            stats,
            dataset_info,
            split_info,
            resume_device=torch.device("cuda"),
        )
        check(
            cuda_compatibility["cuda_rng_state_verified"],
            "strict CUDA resume verifies CUDA RNG state and visible-device topology",
        )
        missing_cuda_rng = dict(loaded)
        missing_cuda_rng["cuda_rng_state"] = None
        try:
            validate_resume_checkpoint(
                missing_cuda_rng,
                args,
                stats,
                dataset_info,
                split_info,
                resume_device=torch.device("cuda"),
            )
        except ValueError:
            rejected_missing_cuda_rng = True
        else:
            rejected_missing_cuda_rng = False
        check(
            rejected_missing_cuda_rng,
            "strict CUDA resume rejects a checkpoint without CUDA RNG continuation state",
        )
        bad_cuda_topology = dict(loaded)
        bad_cuda_topology["cuda_rng_state"] = list(saved_cuda_rng)[:-1]
        try:
            validate_resume_checkpoint(
                bad_cuda_topology,
                args,
                stats,
                dataset_info,
                split_info,
                resume_device=torch.device("cuda"),
            )
        except ValueError:
            rejected_cuda_topology = True
        else:
            rejected_cuda_topology = False
        check(
            rejected_cuda_topology,
            "strict CUDA resume rejects a changed visible-device RNG topology",
        )
    else:
        print("  SKIP CUDA RNG restore/topology gate (CUDA unavailable)")
    bad_stats = {"peak_power": 3.0, "dynamic_range_db": 60.0}
    try:
        validate_resume_checkpoint(loaded, args, bad_stats, dataset_info, split_info)
    except ValueError:
        rejected_mismatch = True
    else:
        rejected_mismatch = False
    check(rejected_mismatch, "checkpoint compatibility rejects a mismatched normalization peak")
    bad_dataset = {**dataset_info, "dataset_path": "/synthetic/other.npz"}
    try:
        validate_resume_checkpoint(loaded, args, stats, bad_dataset, split_info)
    except ValueError:
        rejected_dataset_mismatch = True
    else:
        rejected_dataset_mismatch = False
    check(rejected_dataset_mismatch, "checkpoint compatibility rejects a mismatched dataset identity")
    bad_split = role_provenance([0], [2], [1], test_payload_materialized=True)
    try:
        validate_resume_checkpoint(loaded, args, stats, dataset_info, bad_split)
    except ValueError:
        rejected_split_mismatch = True
    else:
        rejected_split_mismatch = False
    check(rejected_split_mismatch, "strict resume rejects mismatched train/val/test role IDs")
    bad_objective_args = SimpleNamespace(**vars(args))
    bad_objective_args.weight_fft = 0.50
    try:
        validate_resume_checkpoint(loaded, bad_objective_args, stats, dataset_info, split_info)
    except ValueError:
        rejected_objective_mismatch = True
    else:
        rejected_objective_mismatch = False
    check(rejected_objective_mismatch, "strict resume rejects a changed objective weight")
    bad_eval_cadence_args = SimpleNamespace(**vars(args))
    bad_eval_cadence_args.eval_every = 3
    try:
        validate_resume_checkpoint(loaded, bad_eval_cadence_args, stats, dataset_info, split_info)
    except ValueError:
        rejected_eval_cadence_mismatch = True
    else:
        rejected_eval_cadence_mismatch = False
    check(
        rejected_eval_cadence_mismatch,
        "strict resume rejects a changed validation cadence",
    )
    bad_eval_selection_args = SimpleNamespace(**vars(args))
    bad_eval_selection_args.eval_max_views = 2
    try:
        validate_resume_checkpoint(loaded, bad_eval_selection_args, stats, dataset_info, split_info)
    except ValueError:
        rejected_eval_selection_mismatch = True
    else:
        rejected_eval_selection_mismatch = False
    check(
        rejected_eval_selection_mismatch,
        "strict resume rejects a changed validation selection budget",
    )
    incomplete_strict = dict(loaded)
    incomplete_strict["split_provenance"] = None
    try:
        validate_resume_checkpoint(incomplete_strict, args, stats, dataset_info, split_info)
    except ValueError:
        rejected_incomplete_strict = True
    else:
        rejected_incomplete_strict = False
    check(rejected_incomplete_strict, "strict resume rejects missing split provenance")
    missing_state_rejected = []
    for field in (
        "optimizer_state_dict",
        "scheduler_state_dict",
        "numpy_rng_state_json",
        "torch_rng_state",
    ):
        incomplete_state = dict(loaded)
        incomplete_state[field] = None
        try:
            validate_resume_checkpoint(
                incomplete_state, args, stats, dataset_info, split_info
            )
        except ValueError:
            missing_state_rejected.append(field)
    check(
        missing_state_rejected
        == ["optimizer_state_dict", "scheduler_state_dict", "numpy_rng_state_json", "torch_rng_state"],
        "strict resume rejects missing optimizer, scheduler, NumPy RNG, or Torch RNG state",
    )
    legacy_checkpoint = dict(loaded)
    legacy_checkpoint.pop("resume_contract_version")
    legacy_checkpoint["split_provenance"] = None
    legacy = validate_resume_checkpoint(
        legacy_checkpoint, args, stats, dataset_info, split_info
    )
    check(
        legacy["legacy_contract_unverified"] and not legacy["split_provenance_verified"],
        "legacy checkpoint remains resumable but is explicitly split-unverified",
    )
    check(
        provenance["restored"] is True
        and provenance["checkpoint_step"] == 7
        and provenance["checkpoint_file_size_bytes"] > 0,
        "checkpoint provenance records restored identity, training step, and file size",
    )


def _write_sealed_entrypoint_fixture(root: Path):
    """Create a tiny complete-partition NPZ/manifest pair for a CPU entrypoint gate."""

    metadata = {
        "radar_fc_hz": 10.0e9,
        "radar_bandwidth_hz": 3.0e9,
        "num_adc_samples": 4,
    }
    response = np.empty((5, 1, 1, 1, 4), dtype=np.complex64)
    waveform = np.asarray([1.0, 1.0j, -1.0, -1.0j], dtype=np.complex64)
    for view in range(response.shape[0]):
        response[view, 0, 0, 0] = (view + 1) * waveform
    poses = np.zeros((response.shape[0], 3), dtype=np.float32)
    npz_path = root / "sealed_entrypoint.npz"
    np.savez(
        npz_path,
        response=response,
        viewpoint_positions=poses,
        tx_pos=poses[:, None, :],
        rx_pos=poses[:, None, :],
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    manifest = {
        "schema_version": 1,
        "name": "synthetic_radar_fields_sealed_entrypoint_v1",
        "dataset": {
            "path_hint": "/historical/location/that-is-not-the-current-npz-path.npz",
            "response_shape": [5, 1, 1, 1, 4],
            "response_dtype": "complex64",
        },
        "split": {
            "train_indices": [0, 1],
            "validation_indices": [2],
            "test_indices": [3],
            "unused_indices": [4],
            "num_train": 2,
            "num_validation": 1,
            "num_test": 1,
            "num_unused": 1,
            "complete_partition": True,
            "test_sealed": True,
            "unused_sealed": True,
        },
    }
    manifest_path = root / "sealed_entrypoint_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)
    return npz_path, manifest_path, manifest


def _run_radar_fields_entrypoint(argv, *, stop_requested=False):
    """Run the real CLI main with an optional clean-interruption boundary."""

    old_argv = list(sys.argv)
    old_stop = radar_fields_train_entrypoint.STOP_REQUESTED
    try:
        sys.argv = ["train_radar_fields.py", *[str(value) for value in argv]]
        radar_fields_train_entrypoint.STOP_REQUESTED = bool(stop_requested)
        radar_fields_train_entrypoint.main()
    finally:
        sys.argv = old_argv
        radar_fields_train_entrypoint.STOP_REQUESTED = old_stop


def stage_sealed_entrypoint_and_continuation():
    """Exercise the actual RF CLI, lazy boundary, checkpoint, and resume path."""

    print("Stage I: sealed manifest entrypoint and exact continuation")
    with tempfile.TemporaryDirectory(prefix="rift_radar_fields_sealed_entrypoint_") as temporary:
        root = Path(temporary)
        npz_path, manifest_path, manifest = _write_sealed_entrypoint_fixture(root)
        lazy = load_radar_fields_npz(str(npz_path), load_response=False)
        split = load_radar_fields_sealed_split_manifest(
            str(manifest_path),
            5,
            response_shape=lazy._response_shape(),
            response_dtype=lazy.response_dtype,
            expected_num_train=2,
            expected_num_val=1,
            expected_num_test=1,
        )
        check(
            split.train_indices == (0, 1)
            and split.validation_indices == (2,)
            and split.test_indices == (3,)
            and split.unused_indices == (4,),
            "sealed manifest binds explicit ordered roles while treating a historical path hint as provenance only",
        )
        restricted = restrict_radar_fields_response_views(
            lazy, split.train_indices + split.validation_indices
        )
        restricted_provenance = dataset_provenance(restricted)
        try:
            restricted.response_view(split.test_indices[0])
        except PermissionError:
            denied_test = True
        else:
            denied_test = False
        check(
            denied_test
            and not restricted.response_is_materialized
            and restricted_provenance["response_access_restricted"] is True
            and restricted_provenance["authorized_response_view_count"] == 3,
            "sealed reader rejects reserved-test materialization before payload access",
        )
        malformed = dict(manifest)
        malformed["split"] = dict(manifest["split"])
        malformed["split"]["validation_indices"] = [1]
        malformed_path = root / "overlap_manifest.json"
        with open(malformed_path, "w", encoding="utf-8") as handle:
            json.dump(malformed, handle)
        try:
            load_radar_fields_sealed_split_manifest(
                str(malformed_path),
                5,
                response_shape=lazy._response_shape(),
                response_dtype=lazy.response_dtype,
                expected_num_train=2,
                expected_num_val=1,
                expected_num_test=1,
            )
        except ValueError:
            rejected_overlap = True
        else:
            rejected_overlap = False
        check(rejected_overlap, "sealed manifest rejects overlapping source-view roles before response access")

        incomplete_manifest = dict(manifest)
        incomplete_manifest["split"] = dict(manifest["split"])
        incomplete_manifest["split"]["complete_partition"] = False
        incomplete_path = root / "incomplete_manifest.json"
        with open(incomplete_path, "w", encoding="utf-8") as handle:
            json.dump(incomplete_manifest, handle)
        try:
            load_radar_fields_sealed_split_manifest(
                str(incomplete_path),
                5,
                response_shape=lazy._response_shape(),
                response_dtype=lazy.response_dtype,
                expected_num_train=2,
                expected_num_val=1,
                expected_num_test=1,
            )
        except ValueError:
            rejected_incomplete = True
        else:
            rejected_incomplete = False
        check(
            rejected_incomplete,
            "sealed manifest requires an explicit complete partition with sealed unused role",
        )

        wrong_shape_manifest = dict(manifest)
        wrong_shape_manifest["dataset"] = dict(manifest["dataset"])
        wrong_shape_manifest["dataset"]["response_shape"] = [5, 1, 1, 1, 5]
        wrong_shape_path = root / "wrong_shape_manifest.json"
        with open(wrong_shape_path, "w", encoding="utf-8") as handle:
            json.dump(wrong_shape_manifest, handle)
        try:
            load_radar_fields_sealed_split_manifest(
                str(wrong_shape_path),
                5,
                response_shape=lazy._response_shape(),
                response_dtype=lazy.response_dtype,
                expected_num_train=2,
                expected_num_val=1,
                expected_num_test=1,
            )
        except ValueError:
            rejected_wrong_shape = True
        else:
            rejected_wrong_shape = False

        wrong_dtype_manifest = dict(manifest)
        wrong_dtype_manifest["dataset"] = dict(manifest["dataset"])
        wrong_dtype_manifest["dataset"]["response_dtype"] = "complex128"
        wrong_dtype_path = root / "wrong_dtype_manifest.json"
        with open(wrong_dtype_path, "w", encoding="utf-8") as handle:
            json.dump(wrong_dtype_manifest, handle)
        try:
            load_radar_fields_sealed_split_manifest(
                str(wrong_dtype_path),
                5,
                response_shape=lazy._response_shape(),
                response_dtype=lazy.response_dtype,
                expected_num_train=2,
                expected_num_val=1,
                expected_num_test=1,
            )
        except ValueError:
            rejected_wrong_dtype = True
        else:
            rejected_wrong_dtype = False
        check(
            rejected_wrong_shape and rejected_wrong_dtype and not lazy.response_is_materialized,
            "sealed manifest rejects a mismatched full response shape or dtype before payload access",
        )

        # A legacy cache can contain the right train IDs while still having
        # estimated its peak from a reserved response.  In sealed mode that is
        # not reusable: an existing unsafe artifact fails closed without a
        # stream or overwrite, while a missing cache can be created safely.
        unsafe_stats_path = root / "unsafe_sealed_stats.json"
        unsafe_stats = {
            "peak_power": 123.0,
            "dynamic_range_db": 60.0,
            "train_view_indices": [0, 1],
            "train_view_count": 2,
            "normalization_scan_view_indices": [3],
        }
        with open(unsafe_stats_path, "w", encoding="utf-8") as handle:
            json.dump(unsafe_stats, handle)
        observed_stats_batches = []
        original_stats_batch = radar_fields_dataset_module._iter_response_views_from_npz

        def guarded_stats_batch(path, view_indices, shape, dtype):
            requested = tuple(int(value) for value in view_indices)
            observed_stats_batches.append(requested)
            if not set(requested).issubset({0, 1}):
                raise AssertionError(
                    f"sealed normalization attempted non-train response IDs {requested}"
                )
            yield from original_stats_batch(path, requested, shape, dtype)

        radar_fields_dataset_module._iter_response_views_from_npz = guarded_stats_batch
        try:
            try:
                load_or_create_stats(
                    str(unsafe_stats_path),
                    restricted,
                    split.train_indices,
                    dynamic_range_db=60.0,
                    max_views=1,
                    sealed_protocol=True,
                )
            except ValueError:
                rejected_unsafe_stats = True
            else:
                rejected_unsafe_stats = False

            fresh_stats_path = root / "fresh_sealed_stats.json"
            fresh_stats = load_or_create_stats(
                str(fresh_stats_path),
                restricted,
                split.train_indices,
                dynamic_range_db=60.0,
                max_views=1,
                sealed_protocol=True,
            )
        finally:
            radar_fields_dataset_module._iter_response_views_from_npz = original_stats_batch
        with open(unsafe_stats_path, "r", encoding="utf-8") as handle:
            preserved_unsafe_stats = json.load(handle)
        check(
            rejected_unsafe_stats
            and preserved_unsafe_stats == unsafe_stats
            and fresh_stats["normalization_stats_cache_reused"] is False
            and fresh_stats["normalization_provenance_verified"] is True
            and fresh_stats["normalization_scan_view_indices"] == [0]
            and observed_stats_batches == [(0,)],
            "sealed normalization rejects an unsafe cache without a response stream or overwrite; fresh calibration scans only train",
        )

        checkpoint_root = root / "checkpoints"
        common = [
            "--npz-path", npz_path,
            "--checkpoint-root", checkpoint_root,
            "--checkpoint-name", "sealed_cpu",
            "--device", "cpu",
            "--sealed-protocol",
            "--sealed-split-manifest", manifest_path,
            "--num-train", "2",
            "--num-val", "1",
            "--num-test", "1",
            "--steps", "2",
            "--view-batch", "1",
            "--train-pairs", "1",
            "--val-pairs", "1",
            "--eval-every", "1",
            "--checkpoint-every", "1",
            "--eval-max-views", "1",
            "--stats-max-views", "1",
            "--extent", "0.01",
            "--granularity", "2",
            "--hidden-dim", "8",
            "--feature-dim", "4",
            "--sh-degree", "1",
            "--no-batch-norm",
            "--hash-levels", "2",
            "--hash-features", "2",
            "--hash-base-resolution", "4",
            "--hash-final-resolution", "8",
            "--hash-log2-size", "6",
            "--query-chunk", "8",
            "--pair-chunk", "1",
        ]

        def replace_cli_value(argv, flag, value):
            updated = list(argv)
            updated[updated.index(flag) + 1] = value
            return updated

        def remove_cli_option(argv, flag, *, takes_value):
            updated = list(argv)
            position = updated.index(flag)
            del updated[position : position + (2 if takes_value else 1)]
            return updated

        def assert_preflight_rejection(argv, *, expected_load_modes, label):
            """Prove a rejected CLI never reaches stats/model/output work."""

            checkpoint_root_arg = Path(argv[argv.index("--checkpoint-root") + 1])
            checkpoint_name_arg = str(argv[argv.index("--checkpoint-name") + 1])
            rejected_output = checkpoint_root_arg / checkpoint_name_arg
            check(not rejected_output.exists(), f"{label}: rejection fixture starts without an output directory")
            observed_load_modes = []
            observed_stats_calls = []
            observed_model_calls = []
            original_loader = radar_fields_train_entrypoint.load_radar_fields_npz
            original_stats = radar_fields_train_entrypoint.load_or_create_stats
            original_build_model = radar_fields_train_entrypoint.build_model

            def guarded_loader(*args, **kwargs):
                observed_load_modes.append(bool(kwargs.get("load_response", True)))
                return original_loader(*args, **kwargs)

            def guarded_stats(*args, **kwargs):
                observed_stats_calls.append(True)
                raise AssertionError(f"{label}: rejected preflight reached normalization stats")

            def guarded_build_model(*args, **kwargs):
                observed_model_calls.append(True)
                raise AssertionError(f"{label}: rejected preflight reached model construction")

            radar_fields_train_entrypoint.load_radar_fields_npz = guarded_loader
            radar_fields_train_entrypoint.load_or_create_stats = guarded_stats
            radar_fields_train_entrypoint.build_model = guarded_build_model
            try:
                try:
                    _run_radar_fields_entrypoint(argv)
                except ValueError:
                    rejected = True
                else:
                    rejected = False
            finally:
                radar_fields_train_entrypoint.load_radar_fields_npz = original_loader
                radar_fields_train_entrypoint.load_or_create_stats = original_stats
                radar_fields_train_entrypoint.build_model = original_build_model
            check(
                rejected
                and observed_load_modes == list(expected_load_modes)
                and not observed_stats_calls
                and not observed_model_calls
                and not rejected_output.exists(),
                f"{label}: rejected before payload/stats/model work and without an output cache",
            )

        # These execute the real CLI rather than only the manifest helper.  A
        # malformed header binding may inspect metadata/header lazily, but it
        # must never trigger a response materialization, stats write, model,
        # or output directory.
        for malformed_path, suffix in (
            (wrong_shape_path, "wrong_shape"),
            (wrong_dtype_path, "wrong_dtype"),
        ):
            header_rejection_argv = replace_cli_value(
                common, "--sealed-split-manifest", malformed_path
            )
            header_rejection_argv = replace_cli_value(
                header_rejection_argv, "--checkpoint-name", f"header_{suffix}"
            )
            assert_preflight_rejection(
                header_rejection_argv,
                expected_load_modes=[False],
                label=f"sealed CLI {suffix} header binding",
            )

        authorized = {0, 1, 2}
        observed_single = []
        observed_batches = []
        original_single = radar_fields_dataset_module._read_response_view_from_npz
        original_batch = radar_fields_dataset_module._iter_response_views_from_npz

        def guarded_single(path, view_index, shape, dtype):
            view_index = int(view_index)
            observed_single.append(view_index)
            if view_index not in authorized:
                raise AssertionError(f"sealed entrypoint attempted unauthorized response view {view_index}")
            return original_single(path, view_index, shape, dtype)

        def guarded_batch(path, view_indices, shape, dtype):
            requested = tuple(int(value) for value in view_indices)
            observed_batches.append(requested)
            if not set(requested).issubset(authorized):
                raise AssertionError(
                    f"sealed entrypoint attempted unauthorized response batch {requested}"
                )
            yield from original_batch(path, requested, shape, dtype)

        radar_fields_dataset_module._read_response_view_from_npz = guarded_single
        radar_fields_dataset_module._iter_response_views_from_npz = guarded_batch
        try:
            # STOP_REQUESTED models a scheduler-delivered clean interruption
            # after the first actual optimizer update.  The second CLI call
            # must then restore and advance the strict continuation.
            _run_radar_fields_entrypoint(common, stop_requested=True)
            latest = checkpoint_root / "sealed_cpu" / "checkpoint_latest.pth.tar"
            first = torch.load(latest, map_location="cpu", weights_only=False)
            check(
                first["step"] == 1
                and first["split_provenance"]["test_payload_materialized"] is False
                and first["dataset_provenance"]["response_access_restricted"] is True
                and first["sealed_protocol_contract"]["authorized_response_roles"]
                == ["train", "val"]
                and first["sealed_protocol_contract"]["response_header_shape"]
                == [5, 1, 1, 1, 4]
                and first["sealed_protocol_contract"]["response_header_dtype"]
                == "complex64",
                "sealed CLI checkpoint persists an unmaterialized-test protocol contract after a real update",
            )
            _run_radar_fields_entrypoint([*common, "--resume", latest])
            final = checkpoint_root / "sealed_cpu" / "checkpoint_final.pth.tar"
            resumed = torch.load(final, map_location="cpu", weights_only=False)
            check(
                resumed["step"] == 2
                and resumed["sealed_protocol_contract"] == first["sealed_protocol_contract"],
                "sealed CLI resume restores the compatible continuation and completes the next optimizer step",
            )

            sealed_to_legacy_argv = remove_cli_option(
                common, "--sealed-protocol", takes_value=False
            )
            sealed_to_legacy_argv = remove_cli_option(
                sealed_to_legacy_argv, "--sealed-split-manifest", takes_value=True
            )
            sealed_to_legacy_argv = replace_cli_value(
                sealed_to_legacy_argv, "--checkpoint-name", "sealed_to_legacy_preflight"
            )
            assert_preflight_rejection(
                [*sealed_to_legacy_argv, "--resume", final],
                expected_load_modes=[],
                label="sealed checkpoint resumed with legacy flags",
            )

            legacy_like_checkpoint = dict(resumed)
            legacy_like_checkpoint.pop("sealed_protocol_contract")
            legacy_like_path = root / "legacy_like_checkpoint.pth.tar"
            torch.save(legacy_like_checkpoint, legacy_like_path)
            assert_preflight_rejection(
                [
                    *replace_cli_value(common, "--checkpoint-name", "legacy_to_sealed_preflight"),
                    "--resume",
                    legacy_like_path,
                ],
                expected_load_modes=[],
                label="legacy checkpoint resumed with sealed flags",
            )

            incompatible_manifest = dict(manifest)
            incompatible_manifest["split"] = dict(manifest["split"])
            # Same counts, but different explicit role bindings.  A display
            # name/path alone is only provenance and must not act as a pin.
            # Keep training IDs fixed so the cached train-only normalization
            # remains compatible; exchange validation and reserved-test IDs
            # to prove the strict resume contract itself stops the mismatch.
            incompatible_manifest["split"]["validation_indices"] = [3]
            incompatible_manifest["split"]["test_indices"] = [2]
            incompatible_path = root / "incompatible_manifest.json"
            with open(incompatible_path, "w", encoding="utf-8") as handle:
                json.dump(incompatible_manifest, handle)
            incompatible_argv = replace_cli_value(
                common, "--sealed-split-manifest", incompatible_path
            )
            incompatible_argv = replace_cli_value(
                incompatible_argv, "--checkpoint-name", "changed_roles_preflight"
            )
            assert_preflight_rejection(
                [*incompatible_argv, "--resume", final],
                expected_load_modes=[False],
                label="sealed checkpoint resumed with changed role IDs",
            )

            schema_changed_manifest = dict(manifest)
            schema_changed_manifest["schema_version"] = 2
            schema_changed_path = root / "schema_changed_manifest.json"
            with open(schema_changed_path, "w", encoding="utf-8") as handle:
                json.dump(schema_changed_manifest, handle)
            schema_changed_argv = replace_cli_value(
                common, "--sealed-split-manifest", schema_changed_path
            )
            schema_changed_argv = replace_cli_value(
                schema_changed_argv, "--checkpoint-name", "schema_changed_preflight"
            )
            assert_preflight_rejection(
                [*schema_changed_argv, "--resume", final],
                expected_load_modes=[False],
                label="sealed checkpoint resumed with changed manifest schema version",
            )

            # A corrected archive may live at a new path even when a frozen
            # manifest still carries historical provenance.  Relocation alone
            # must not become a path/hash pin when the header and role lists
            # validate again before any response is read.
            relocated_npz = root / "relocated_sealed_entrypoint.npz"
            relocated_manifest = root / "relocated_sealed_entrypoint_manifest.json"
            shutil.copyfile(npz_path, relocated_npz)
            shutil.copyfile(manifest_path, relocated_manifest)
            relocated_argv = list(common)
            relocated_argv[relocated_argv.index("--npz-path") + 1] = relocated_npz
            relocated_argv[
                relocated_argv.index("--sealed-split-manifest") + 1
            ] = relocated_manifest
            _run_radar_fields_entrypoint(
                [*relocated_argv, "--resume", final, "--eval-only"]
            )
            check(
                True,
                "sealed CLI resume accepts an equivalent relocated archive/manifest without a path or hash pin",
            )
        finally:
            radar_fields_dataset_module._read_response_view_from_npz = original_single
            radar_fields_dataset_module._iter_response_views_from_npz = original_batch
        observed = set(observed_single)
        for requested in observed_batches:
            observed.update(requested)
        check(
            observed and observed.issubset(authorized) and 3 not in observed and 4 not in observed,
            "actual sealed train/validation/continuation entrypoints never stream reserved-test or unused responses",
        )


def stage_real_dataset(npz_path, *, num_train, num_val, num_test, seed, val_from_tail):
    """Validate one authorized train view without materializing sealed payloads.

    The real-data gate intentionally shares the ordinary Radar Fields split
    convention.  It obtains all role identities from pose/header metadata,
    then streams exactly one train-role response.  In particular it never
    falls back to raw source view zero, which could be a held-out view.
    """

    print("Stage J: existing RIFT dataset contract (role-restricted lazy read)")
    arrays = load_radar_fields_npz(npz_path, load_response=False)
    check(
        arrays.response is None
        and not arrays.response_is_materialized
        and arrays.source_path is not None
        and arrays.response_shape is not None,
        "real-data gate resolves header and poses without materializing the response payload",
    )
    check(
        arrays.tx_pos.shape == (arrays.num_views, arrays.num_tx, 3)
        and arrays.rx_pos.shape == (arrays.num_views, arrays.num_rx, 3),
        "measured Tx/Rx positions match the response cube",
    )
    check(
        arrays.viewpoint_positions.shape == (arrays.num_views, 3),
        "viewpoint positions match the response cube",
    )
    train_indices, val_indices, test_indices = split_view_indices(
        arrays.num_views,
        num_train,
        num_val,
        num_test,
        seed,
        val_from_tail=val_from_tail,
    )
    authorized_indices = np.concatenate((train_indices, val_indices))
    arrays = restrict_radar_fields_response_views(arrays, authorized_indices)
    selected_view = resolve_diagnostic_view("train", 0, train_indices, val_indices)
    roles = role_provenance(
        train_indices,
        val_indices,
        test_indices,
        test_payload_materialized=False,
    )
    check(
        selected_view == int(train_indices[0])
        and roles["test_payload_materialized"] is False,
        "real-data gate selects an explicit canonical train-role source view before streaming",
    )
    authorized_set = {int(value) for value in authorized_indices}
    if len(test_indices):
        denied_view = int(test_indices[0])
        denied_label = "test"
    else:
        # Older optional diagnostic invocations sometimes configured no formal
        # test role.  Still prove the capability cannot open any unassigned
        # source response rather than silently treating an unrestricted handle
        # as role-restricted.
        denied_view = next(
            (index for index in range(arrays.num_views) if index not in authorized_set),
            None,
        )
        denied_label = "unassigned"
    if denied_view is not None:
        try:
            arrays.response_view(denied_view)
        except PermissionError:
            denied_reserved = True
        else:
            denied_reserved = False
        check(
            denied_reserved and denied_view not in authorized_set,
            f"real-data gate rejects the selected {denied_label} source response before payload access",
        )
    else:
        check(
            arrays.response_access_is_restricted
            and len(authorized_set) == arrays.num_views,
            "real-data gate retains an explicit capability even when its legacy roles cover every source view",
        )
    power = response_view_to_range_power(arrays.response_view(selected_view), pair_indices=[0])
    check(
        power.shape == (1, arrays.num_freq) and torch.isfinite(power).all() and (power >= 0).all(),
        "the authorized real train view converts to finite nonnegative range power",
    )
    check(
        arrays.response is None and not arrays.response_is_materialized,
        "streaming the authorized view does not retain the full response or test payload",
    )
    check(range_bin_size(arrays.metadata) > 0, "real dataset metadata defines a positive range-bin spacing")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", default=None, help="pinned official RadarFields checkout")
    parser.add_argument(
        "--npz-path",
        default=None,
        help="optional existing RIFT NPZ role-restricted lazy-read contract gate",
    )
    parser.add_argument(
        "--real-num-train",
        type=int,
        default=1800,
        help="canonical training-view count used only by the --npz-path contract gate",
    )
    parser.add_argument(
        "--real-num-val",
        type=int,
        default=200,
        help="canonical validation-view count used only by the --npz-path contract gate",
    )
    parser.add_argument(
        "--real-num-test",
        type=int,
        default=0,
        help="canonical held-out-view count used only by the --npz-path contract gate",
    )
    parser.add_argument(
        "--real-seed",
        type=int,
        default=42,
        help="canonical split seed used only by the --npz-path contract gate",
    )
    parser.add_argument(
        "--real-val-from-tail",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="canonical validation split convention used only by the --npz-path contract gate",
    )
    args = parser.parse_args()
    stage_intensity(args.upstream)
    stage_ifft()
    stage_projector()
    stage_model_and_loss()
    stage_padded_roi_diagnostic()
    stage_metadata_and_lazy_reader()
    stage_geometry_compatibility()
    stage_checkpoint_roundtrip()
    stage_sealed_entrypoint_and_continuation()
    if args.npz_path:
        stage_real_dataset(
            args.npz_path,
            num_train=args.real_num_train,
            num_val=args.real_num_val,
            num_test=args.real_num_test,
            seed=args.real_seed,
            val_from_tail=args.real_val_from_tail,
        )
    else:
        print("Stage J: SKIP existing RIFT dataset contract (pass --npz-path to require it)")
    print(f"\nALL RADAR FIELDS GATES PASSED ({CHECKS} checks)")


if __name__ == "__main__":
    main()
