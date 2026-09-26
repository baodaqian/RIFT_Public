#!/usr/bin/env python
"""Fail-closed CPU validation for the fresh GOTCHA full-domain v1 trainer."""

from __future__ import annotations

import argparse
import copy
import gc
import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_trainer():
    path = PROJECT_ROOT / "scripts" / "public_radar_gotcha_full_domain_v1_partial.py"
    spec = importlib.util.spec_from_file_location(
        "validate_public_radar_gotcha_full_domain_v1_trainer", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expect_value_error(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError(f"{function.__name__} did not reject invalid state")


def checkpoint_identity_gate(trainer):
    model = torch.nn.Linear(2, 2, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-5, eps=1.0e-15)
    cell = "gotcha_p2_full_domain_bp400_deg0"
    contract = {
        "schema": trainer.SCHEMA,
        "schema_version": trainer.SCHEMA_VERSION,
        "cell": cell,
        "shape": [1776, 1776, 1],
    }
    checkpoint = trainer.checkpoint_payload(
        cell,
        0,
        0,
        model,
        optimizer,
        1.0 + 0.0j,
        [],
        contract,
    )
    gain = trainer.validate_checkpoint_identity(
        checkpoint,
        cell=cell,
        degree=0,
        contract=contract,
        model=model,
        context="validation",
    )
    assert gain == 1.0 + 0.0j

    old_schema = copy.deepcopy(checkpoint)
    old_schema["schema"] = "rift.public_radar_interpolation_dense_v2"
    expect_value_error(
        trainer.validate_checkpoint_identity,
        old_schema,
        cell=cell,
        degree=0,
        contract=contract,
        model=model,
        context="old-schema",
    )
    other_cell = copy.deepcopy(checkpoint)
    other_cell["cell"] = "gotcha_p2_full_domain_bp1600_deg0"
    expect_value_error(
        trainer.validate_checkpoint_identity,
        other_cell,
        cell=cell,
        degree=0,
        contract=contract,
        model=model,
        context="other-cell",
    )
    old_shape = copy.deepcopy(checkpoint)
    key = next(iter(old_shape["model_state_dict"]))
    old_shape["model_state_dict"][key] = torch.zeros(1)
    expect_value_error(
        trainer.validate_checkpoint_identity,
        old_shape,
        cell=cell,
        degree=0,
        contract=contract,
        model=model,
        context="old-shape",
    )
    wrong_dtype = copy.deepcopy(checkpoint)
    key = next(iter(wrong_dtype["model_state_dict"]))
    wrong_dtype["model_state_dict"][key] = wrong_dtype["model_state_dict"][key].double()
    expect_value_error(
        trainer.validate_checkpoint_identity,
        wrong_dtype,
        cell=cell,
        degree=0,
        contract=contract,
        model=model,
        context="wrong-dtype",
    )
    other_contract = copy.deepcopy(checkpoint)
    other_contract["contract"]["shape"] = [1024, 1024, 1]
    expect_value_error(
        trainer.validate_checkpoint_identity,
        other_contract,
        cell=cell,
        degree=0,
        contract=contract,
        model=model,
        context="old-contract",
    )


def directory_mode_gate(trainer):
    contract = {
        "schema": trainer.SCHEMA,
        "schema_version": trainer.SCHEMA_VERSION,
        "cell": "gotcha_p2_full_domain_bp400_deg0",
    }
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fresh = root / "fresh"
        trainer.prepare_run_directory(fresh, "fresh", contract)
        assert json.loads((fresh / "contract.json").read_text()) == contract
        expect_value_error(
            trainer.prepare_run_directory, fresh, "fresh", contract
        )
        trainer.prepare_run_directory(fresh, "resume", contract)
        expect_value_error(
            trainer.prepare_run_directory,
            fresh,
            "resume",
            {**contract, "cell": "gotcha_p2_full_domain_bp1600_deg0"},
        )

        empty = root / "empty"
        empty.mkdir()
        trainer.prepare_run_directory(empty, "resume", contract)
        assert (empty / "contract.json").is_file()

        orphan = root / "orphan"
        orphan.mkdir()
        (orphan / "initialization.json").write_text("{}\n", encoding="utf-8")
        expect_value_error(
            trainer.prepare_run_directory, orphan, "resume", contract
        )


class FakeRenderer:
    def __init__(self, fail_on=None):
        self.freqs = torch.arange(3, dtype=torch.float64)
        self.num_rx = 1
        self.num_tx = 1
        self.fail_on = fail_on

    def raw_prediction_and_measurement(self, model, item):
        if self.fail_on is not None and item == self.fail_on:
            raise RuntimeError("forced evaluation interruption")
        base = float(item + 1)
        predicted = torch.tensor(
            [base + 0.5j, 0.5 * base - 0.25j, -0.25 * base + 0.75j],
            dtype=torch.complex128,
        ).reshape(3, 1, 1)
        measured = torch.tensor(
            [0.8 * base - 0.1j, 0.3 * base + 0.2j, -0.1 * base + 0.4j],
            dtype=torch.complex128,
        ).reshape(3, 1, 1)
        return predicted, measured


def evaluation_resume_gate(trainer):
    model = torch.nn.Identity()
    dataset = list(range(7))
    contract = {
        "schema": trainer.SCHEMA,
        "schema_version": trainer.SCHEMA_VERSION,
        "cell": "gotcha_p2_full_domain_bp400_deg0",
    }
    cell = contract["cell"]
    old_interval = trainer.EVALUATION_CHECKPOINT_EVERY_VIEWS
    trainer.EVALUATION_CHECKPOINT_EVERY_VIEWS = 2
    try:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference, _ = trainer.resumable_fixed_evaluate(
                model,
                0.75 - 0.2j,
                dataset,
                FakeRenderer(),
                cell=cell,
                contract=contract,
                epoch_number=1,
                split="train",
                state_path=root / "reference.json",
            )
            interrupted_path = root / "interrupted.json"
            try:
                trainer.resumable_fixed_evaluate(
                    model,
                    0.75 - 0.2j,
                    dataset,
                    FakeRenderer(fail_on=2),
                    cell=cell,
                    contract=contract,
                    epoch_number=1,
                    split="train",
                    state_path=interrupted_path,
                )
            except RuntimeError as exc:
                assert "forced evaluation interruption" in str(exc)
            else:
                raise AssertionError("forced evaluation interruption did not occur")
            partial = json.loads(interrupted_path.read_text(encoding="utf-8"))
            assert partial["next_view_index"] == 2
            resumed, _ = trainer.resumable_fixed_evaluate(
                model,
                0.75 - 0.2j,
                dataset,
                FakeRenderer(),
                cell=cell,
                contract=contract,
                epoch_number=1,
                split="train",
                state_path=interrupted_path,
                initial_state=partial,
            )
            for key in (
                "relative_mse",
                "relative_l2",
                "coherent_correlation",
                "sample_count",
                "raw_cross_real",
                "raw_cross_imag",
                "raw_prediction_power_sum",
                "target_power_sum",
            ):
                assert resumed[key] == reference[key], (key, resumed[key], reference[key])

            corrupt = json.loads(interrupted_path.read_text(encoding="utf-8"))
            corrupt["sample_count"] += 1
            expect_value_error(
                trainer._validate_evaluation_state,
                corrupt,
                cell=cell,
                contract=contract,
                epoch_number=1,
                split="train",
                dataset_size=len(dataset),
                samples_per_view=3,
            )
    finally:
        trainer.EVALUATION_CHECKPOINT_EVERY_VIEWS = old_interval


def aligned_renderer_gate(trainer):
    """The trainer's collation and aligned render must equal scalar calls."""
    torch.manual_seed(2301)
    frequencies = np.linspace(9.0e9, 9.2e9, 17, dtype=np.float64)
    platforms = torch.tensor(
        [[7.0, 4.0, 3.0], [7.2, 3.9, 3.1]], dtype=torch.float64
    )
    arrays = {
        "frequencies_hz": frequencies,
        "rx_pos": platforms[:, None, :].numpy(),
        "tx_pos": platforms[:, None, :].numpy(),
    }
    metadata = {
        "propagation_model": "monostatic_near_field_reference",
        "reference_range_m": 7.5,
        "scene_center_m": [0.0, 0.0, 0.0],
    }
    renderer = trainer.SerializedRenderer(
        arrays, metadata, torch.device("cpu"), point_chunk=3, pair_chunk=1
    )
    model = trainer.PLANAR.FixedPlanarSHScene(
        5, 4, 0.4, 3, torch.device("cpu"), init_scale=0.02
    )
    items = []
    for index in range(2):
        magnitude = torch.linspace(
            0.8 + 0.1 * index, 1.2 + 0.1 * index, frequencies.size
        )
        phase = torch.linspace(-0.4 + 0.1 * index, 0.5, frequencies.size)
        items.append(
            (
                index,
                torch.tensor([0.7 + 0.2 * index], dtype=torch.float32),
                torch.tensor([0.3 + 0.1 * index], dtype=torch.float32),
                magnitude,
                phase,
                platforms[index].reshape(1, 3),
                platforms[index].reshape(1, 3),
            )
        )
    scalar = [
        renderer.raw_prediction_and_measurement(model, item) for item in items
    ]
    expected_prediction = torch.stack([value[0] for value in scalar])
    expected_measurement = torch.stack([value[1] for value in scalar])
    actual_prediction, actual_measurement = (
        renderer.raw_prediction_and_measurement_batch(model, items)
    )
    assert actual_prediction.shape == (2, frequencies.size, 1, 1)
    assert actual_measurement.shape == (2, frequencies.size, 1, 1)
    prediction_relative_error = float(
        torch.linalg.vector_norm(actual_prediction - expected_prediction)
        / torch.linalg.vector_norm(expected_prediction).clamp_min(1.0e-30)
    )
    print(
        "INFO: aligned renderer legacy-GEMV/batched-GEMM prediction "
        f"relative_error={prediction_relative_error:.9g}",
        flush=True,
    )
    assert prediction_relative_error <= 5.0e-6
    assert torch.equal(actual_measurement, expected_measurement)


def real_archive_gate(trainer, npz_path):
    arrays, metadata, partition, support = trainer.load_dataset_contract(
        npz_path, "gotcha_p2_full_domain"
    )
    assert tuple(partition[role] for role in ("train", "validation", "test")) == (
        33_939,
        4_242,
        4_243,
    )
    provenance = support["archive_provenance"]
    assert provenance["frequencies_dtype"] == "float64"
    assert provenance["response_dtype"] == "complex64"
    assert provenance["first_frequency_hz"] == 9_288_080_384.0
    assert provenance["last_frequency_hz"] == 9_910_448_128.0
    assert support["preflight"]["view_count"] == 42_424
    assert support["preflight"]["mask_realization"] == (
        "identity_after_exact_planar_extrema_preflight"
    )
    geometry = support["geometry_provenance"]
    assert geometry["preflight_positions"] == "monostatic_tx_rx_phase_centers"
    assert geometry["tx_rx_max_offset_m"] <= geometry["tx_rx_tolerance_m"]
    assert geometry["phase_center_viewpoint_max_offset_m"] <= (
        geometry["phase_center_viewpoint_tolerance_m"]
    )
    selections = {}
    for count in (400, 1600):
        selected, selection = trainer.select_bp_views(
            arrays, "gotcha_p2_full_domain", count
        )
        selections[count] = np.asarray(selected, dtype=np.int64)
        assert len(selected) == count
        assert selection["unique_groups"] == 288
    assert np.array_equal(selections[400], selections[1600][:400])

    for degree, basis_count in ((0, 1), (3, 16)):
        model = trainer.make_scene("gotcha_p2_full_domain", degree, torch.device("cpu"))
        assert model.n_points == 3_154_176
        assert model.w_re.shape == (3_154_176, basis_count)
        assert model.w_im.shape == (3_154_176, basis_count)
        first = model.position_chunk(0, 1)
        last = model.position_chunk(model.n_points - 1, model.n_points)
        assert first.shape == (1, 3) and last.shape == (1, 3)
        assert float(first[0, 2]) == 0.0 and float(last[0, 2]) == 0.0
        assert bool((first[0, :2] > -50.0).all())
        assert bool((last[0, :2] < 50.0).all())
        del model
        gc.collect()

    source = (PROJECT_ROOT / "scripts" / "public_radar_gotcha_full_domain_v1_partial.py").read_text(
        encoding="utf-8"
    )
    assert "test_dataset" not in source
    assert "test_evaluated\": False" in source
    assert metadata["split_interpolation_only"] is True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz-path", required=True)
    args = parser.parse_args()
    trainer = load_trainer()

    expected_cells = {
        f"gotcha_p2_full_domain_bp{bp}_deg{degree}"
        for bp in (400, 1600)
        for degree in (0, 3)
    }
    for cell in expected_cells:
        trainer.parse_cell(cell)
    for retired in (
        "gotcha_p2_bp400_deg0",
        "gotcha_p2_bp1600_deg3",
        "camry_bp400_deg0",
    ):
        expect_value_error(trainer.parse_cell, retired)

    checkpoint_identity_gate(trainer)
    directory_mode_gate(trainer)
    evaluation_resume_gate(trainer)
    aligned_renderer_gate(trainer)
    real_archive_gate(trainer, args.npz_path)
    print("GOTCHA_FULL_DOMAIN_V1_VALIDATION_OK", flush=True)


if __name__ == "__main__":
    main()
