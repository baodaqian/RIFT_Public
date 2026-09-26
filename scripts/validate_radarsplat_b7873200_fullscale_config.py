#!/usr/bin/env python
"""Torch-free validation of the reviewed RadarSplat full-scale candidate."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_NPZ = "/storage/home/hcoda1/1/dbao31/r-jromberg3-0/RIFT/data/b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz"
CANONICAL_MANIFEST = "/storage/scratch1/1/dbao31/rift_round8b_impl_20260810/splits/round8b/b78710k_interp_seed42_train3200_val1000_test1000_v1.json"


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise AssertionError(f"{label} must be a mapping")
    return value


def _read_json(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return _mapping(json.load(handle), str(path))


def _source(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    ast.parse(text, filename=str(path))
    return text


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _assert_equal(observed: object, expected: object, label: str) -> None:
    if observed != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {observed!r}")


def _line_prefix(path: Path, prefix: str) -> bool:
    if not path.is_file():
        return False
    return any(line.startswith(prefix) for line in path.read_text(encoding="utf-8").splitlines())


def _text_prefix(text: str, prefix: str) -> bool:
    return any(line.startswith(prefix) for line in text.splitlines())


def _pipeline_failure_code(child_status: int, *tee_statuses: int) -> int:
    """Return the first nonzero status in launcher pipeline order."""
    if child_status != 0:
        return child_status
    for status in tee_statuses:
        if status != 0:
            return status
    return 0


def _validated_signal_code(child_status: int, *tee_statuses: int) -> int:
    """Mirror clean-stop handling, rejecting a clean child plus tee failure."""
    if child_status == 143:
        return 143 if all(status == 0 for status in tee_statuses) else 96
    return _pipeline_failure_code(child_status, *tee_statuses)


def _fixture_pipeline_return_codes() -> None:
    """Exercise actual status control flow, not marker presence alone."""
    cases = {
        "prep success": ((0, 0), 0),
        "prep child failure": ((7, 0), 7),
        "prep tee failure": ((0, 9), 9),
        "fit success": ((0, 0, 0), 0),
        "fit child failure": ((11, 0, 0), 11),
        "fit first tee failure": ((0, 13, 0), 13),
        "fit second tee failure": ((0, 0, 17), 17),
        "readout success": ((0, 0), 0),
        "readout child failure": ((19, 0), 19),
        "readout tee failure": ((0, 23), 23),
    }
    for label, (statuses, expected) in cases.items():
        observed = _pipeline_failure_code(*statuses)
        if observed != expected:
            raise AssertionError(f"{label}: expected {expected}, got {observed}")
    signal_cases = {
        "prep child 143 and clean tee": ((143, 0), 143),
        "prep child 143 and tee failure": ((143, 29), 96),
        "fit child 143 and clean tees": ((143, 0, 0), 143),
        "fit child 143 and tee failure": ((143, 0, 31), 96),
        "readout child 143 and clean tee": ((143, 0), 143),
        "readout child 143 and tee failure": ((143, 37), 96),
    }
    for label, (statuses, expected) in signal_cases.items():
        observed = _validated_signal_code(*statuses)
        if observed != expected:
            raise AssertionError(f"{label}: expected {expected}, got {observed}")


def _fixture_lifecycle_checks() -> None:
    """Exercise launcher marker decisions without importing Torch or running Slurm."""
    prepare = "RADARSPLAT_B7873200_PREPARE_CLEAN_STOP_RETAINED signal=15 materialized=3 reused=0\n"
    if not _text_prefix(prepare, "RADARSPLAT_B7873200_PREPARE_CLEAN_STOP_RETAINED"):
        raise AssertionError("partial preparation clean-stop prefix was not accepted")
    prepare = "ERROR: terminal preparation failure\n"
    if _text_prefix(prepare, "RADARSPLAT_B7873200_PREPARE_CLEAN_STOP_RETAINED"):
        raise AssertionError("arbitrary preparation failure was accepted as resumable")
    prepare = "RADARSPLAT_B7873200_TARGET_CACHE_COMPLETE\n"
    if not _text_prefix(prepare, "RADARSPLAT_B7873200_TARGET_CACHE_COMPLETE"):
        raise AssertionError("completed-cache-before-fit marker was not accepted")

    fit = "RADARSPLAT_B7873200_FULLSCALE_CLEAN_STOP_RETAINED\n"
    if not _text_prefix(fit, "RADARSPLAT_B7873200_FULLSCALE_CLEAN_STOP_RETAINED"):
        raise AssertionError("clean-fit continuation marker was not accepted")
    fit = "ERROR: terminal fit failure\n"
    if _text_prefix(fit, "RADARSPLAT_B7873200_FULLSCALE_CLEAN_STOP_RETAINED"):
        raise AssertionError("terminal fit failure was accepted as resumable")

    readout = "ERROR: terminal readout failure\n"
    if _text_prefix(readout, "RADARSPLAT_B7873200_READOUT_CLEAN_STOP"):
        raise AssertionError("terminal readout failure was accepted as resumable")
    readout = "RADARSPLAT_B7873200_READOUT_CLEAN_STOP\n"
    output_exists = False
    if not _text_prefix(readout, "RADARSPLAT_B7873200_READOUT_CLEAN_STOP") or output_exists:
        raise AssertionError("clean readout continuation fixture was not accepted")
    output_exists = True
    if not output_exists:
        raise AssertionError("published readout was not treated as a resume blocker")

    completion = "RadarSplat B7873200 completion artifacts are present; no update executed.\n"
    if not _text_prefix(completion, "RadarSplat B7873200 completion artifacts are present; no update executed."):
        raise AssertionError("zero-update terminal completion resume was not accepted")


def _fixture_best_checkpoint_selection() -> None:
    """Exercise earliest-tie selection and selected-state provenance."""

    history = [
        {"step": 600, "relative_mse_native_power": 0.50, "active_sh_degree": 0},
        {"step": 1200, "relative_mse_native_power": 0.25, "active_sh_degree": 1},
        {"step": 1800, "relative_mse_native_power": 0.25, "active_sh_degree": 2},
        {"step": 384000, "relative_mse_native_power": 0.30, "active_sh_degree": 3},
        {"step": 480000, "relative_mse_native_power": 0.30, "active_sh_degree": 3},
    ]
    selected_index = None
    selected_metric = None
    for index, row in enumerate(history):
        metric = float(row["relative_mse_native_power"])
        if selected_metric is None or metric < selected_metric:
            selected_index = index
            selected_metric = metric
    if selected_index != 1 or selected_metric != 0.25:
        raise AssertionError("best-checkpoint fixture did not select the first minimum")
    if history[2]["relative_mse_native_power"] == selected_metric and selected_index != 1:
        raise AssertionError("an exact later tie replaced the earliest selected checkpoint")

    checkpoints = {
        1200: {"state": "selected", "geometry": "gaussian_occupancy_geometry_selected.npz"},
        480000: {"state": "final", "geometry": "gaussian_occupancy_geometry.npz"},
    }
    selected = checkpoints[history[selected_index]["step"]]
    terminal = checkpoints[history[-1]["step"]]
    if selected["state"] != "selected" or selected["geometry"] == terminal["geometry"]:
        raise AssertionError("selected metrics/geometry provenance silently came from final state")

    running_best = {}
    for endpoint in (384000, 480000):
        prefix = [row for row in history if row["step"] <= endpoint]
        endpoint_best_index = None
        endpoint_best_metric = None
        for index, row in enumerate(prefix):
            metric = float(row["relative_mse_native_power"])
            if endpoint_best_metric is None or metric < endpoint_best_metric:
                endpoint_best_index = index
                endpoint_best_metric = metric
        if endpoint_best_index != 1 or endpoint_best_metric != 0.25:
            raise AssertionError("running-best budget fixture did not preserve earliest minimum")
        running_best[endpoint] = {
            "step": endpoint,
            "running_best_validation_relative_mse_native_power": endpoint_best_metric,
            "selected_validation_step": prefix[endpoint_best_index]["step"],
        }
    relative_improvement = (0.25 - 0.25) / 0.25
    budget_status = {
        "step_384000": running_best[384000],
        "step_480000": running_best[480000],
        "relative_improvement": relative_improvement,
        "threshold": 0.01,
        "label": "<1% best-validation improvement over the last 30 passes",
        "descriptive_only": True,
        "automatic_extension": False,
    }
    if budget_status["relative_improvement"] != 0.0 or not budget_status["descriptive_only"] or budget_status["automatic_extension"]:
        raise AssertionError("budget-status fixture changed the descriptive fixed-budget policy")

    report = {
        "target_provenance": {
            "dataset_npz_path": "/canonical/b787.npz",
            "role_manifest_path": "/canonical/manifest.json",
            "role_manifest_name": "b78710k_interp_seed42_train3200_val1000_test1000_v1",
            "train_view_ids": [3, 1],
            "validation_view_ids": [8],
            "target_axes": ["azimuth", "range"],
            "projection": "sum_elevation(abs(matched_filter_complex)**2) -> [azimuth,range]",
            "grid_crop": {"n_azimuth": 32, "n_range": 32},
            "matched_filter": {"response_layout": "tx_rx_freq"},
            "normalization": {"mode": "linear_peak", "fit_split": "train", "clip": False, "train_peak_power": 2.0},
        },
        "learning_curve_budget_status": budget_status,
    }
    if set(report["target_provenance"]) != {
        "dataset_npz_path", "role_manifest_path", "role_manifest_name", "train_view_ids",
        "validation_view_ids", "target_axes", "projection", "grid_crop", "matched_filter", "normalization",
    }:
        raise AssertionError("target-provenance fixture lacks required fields")
    if report["learning_curve_budget_status"]["step_384000"]["selected_validation_step"] != 1200:
        raise AssertionError("budget status did not expose earliest running-best selection provenance")


def validate(config_path: Path, launcher_path: Path) -> None:
    config = _read_json(config_path)
    _assert_equal(config.get("schema"), "rift_radarsplat_b7873200_fullscale_candidate_v1", "schema")
    _assert_equal(config.get("status"), "review_candidate_unexecuted", "status")
    _assert_equal(config.get("production_clearance"), False, "production clearance")

    dataset = _mapping(config.get("dataset"), "dataset")
    _assert_equal(dataset.get("npz_path"), CANONICAL_NPZ, "canonical archive")
    _assert_equal(dataset.get("role_manifest"), CANONICAL_MANIFEST, "canonical manifest")
    _assert_equal(dataset.get("response_shape"), [10000, 16, 16, 1, 600], "response shape")
    _assert_equal(dataset.get("response_dtype"), "complex64", "response dtype")
    _assert_equal(dataset.get("coordinates_per_view"), "16 Tx x 16 Rx x 600 frequency samples", "coordinate coverage")

    split = _mapping(config.get("split"), "split")
    for key, expected in (("seed", 42), ("train_count", 3200), ("validation_count", 1000), ("reserved_test_count", 1000), ("unused_count", 4800)):
        _assert_equal(split.get(key), expected, key)
    _assert_equal(split.get("fit_materialized_roles"), ["train", "validation"], "fit roles")
    _assert_equal(split.get("reserved_test_materialized"), False, "test sealing")

    target = _mapping(config.get("target"), "target")
    _assert_equal(target.get("observable"), "native real local-polar power", "observable")
    _assert_equal(target.get("projection"), "sum_elevation(abs(matched_filter_complex)**2)", "native projection")
    _assert_equal(target.get("coherent_phase_target"), False, "coherent target policy")
    _assert_equal(target.get("backend"), "range_nufft", "target backend")
    _assert_equal(target.get("compute_dtype"), "float64", "target dtype")
    matched_filter = _mapping(target.get("matched_filter"), "matched filter")
    for key, expected in (("phase_sign", -1.0), ("response_layout", "tx_rx_freq"), ("range_model", "none"), ("include_four_pi", False)):
        _assert_equal(matched_filter.get(key), expected, f"matched filter {key}")
    grid = _mapping(target.get("grid"), "target grid")
    for key, expected in (("scene_extent_m", 0.15), ("n_azimuth", 32), ("n_elevation", 32), ("n_range", 32), ("output_azimuth_resolution_deg", 0.9), ("elevation_sampling_resolution_deg", 0.9), ("intermediate_azimuth_resolution_deg", 0.09)):
        _assert_equal(grid.get(key), expected, f"grid {key}")

    model = _mapping(config.get("model"), "model")
    for key, expected in (("initial_gaussians", 2048), ("init_extent_m", 0.3), ("init_scale_m", 0.003), ("init_opacity", 0.1), ("init_noise_probability", 0.1), ("sh_degree", 3), ("sh_degree_interval", 600), ("planar_initialization", True)):
        _assert_equal(model.get(key), expected, f"model {key}")
    optimization = _mapping(config.get("optimization"), "optimization")
    for key, expected in (("steps", 480000), ("epochs", 150), ("updates_per_epoch", 3200), ("validation_every", 600), ("checkpoint_every", 100), ("log_every", 1)):
        _assert_equal(optimization.get(key), expected, f"optimization {key}")
    strategy = _mapping(config.get("strategy"), "strategy")
    if "disabled" not in str(strategy.get("densification")):
        raise AssertionError("the candidate must disclose that densification is not implemented")
    _assert_equal(strategy.get("prune_opacity"), 0.0005, "prune opacity")
    _assert_equal(strategy.get("prune_every"), 100, "prune schedule")
    renderer = _mapping(config.get("renderer"), "renderer")
    _assert_equal(renderer.get("backend"), "torch_sparse_reference", "renderer backend")
    _assert_equal(renderer.get("gaussian_chunk_size"), 64, "renderer chunk")
    _assert_equal(renderer.get("max_raster_candidate_pairs"), 2000000, "raster pair cap")
    resources = _mapping(config.get("resources"), "resources")
    for key, expected in (("qos", "inferno"), ("partition", "gpu-rtx6000"), ("gpus", 1), ("cpus", 6), ("memory_gib", 32), ("temporary_gib", 12), ("time_limit", "12:00:00"), ("signal_seconds", 300)):
        _assert_equal(resources.get(key), expected, f"resource {key}")

    launcher = _text(launcher_path)
    required_fragments = (
        "#SBATCH --qos=inferno",
        "#SBATCH --partition=gpu-rtx6000",
        "#SBATCH --gres=gpu:rtx_6000:1",
        "#SBATCH --cpus-per-task=6",
        "#SBATCH --mem=32G",
        "#SBATCH --time=12:00:00",
        "#SBATCH --signal=TERM@300",
        "#SBATCH --export=NONE",
        '[[ "${SLURM_JOB_QOS:-}" == "inferno" && "${SLURM_JOB_PARTITION:-}" == "gpu-rtx6000" ]]',
        "RUN_STEP=(srun --nodes=1 --ntasks=1 --cpus-per-task=6",
        "--signal=TERM@300",
        "has_attempt_prefix",
        "has_line_prefix",
        "has_attempt_file",
        "--npz-path \"$NPZ\"",
        "--role-manifest \"$PARENT_MANIFEST\"",
        "--max-train 0 --max-validation 0",
        "--steps 480000",
        "--epochs 150",
        "--validation-every 600",
        "--checkpoint-every 100",
        "--log-every 1",
        "--init-num-gaussians 2048",
        "--sh-degree-interval 600",
        "--prune-every 100",
        "--gaussian-chunk-size 64",
        "--max-raster-candidate-pairs 2000000",
        "pipeline_failure_code",
        "RADARSPLAT_B7873200_SIGNAL_PROBE_START",
        "RADARSPLAT_B7873200_SIGNAL_PROBE_PASS",
        "signal_probe_status",
        '"/usr/bin/time" -v -o "$prepare_time_log" "${RUN_STEP[@]}" python',
        "scripts/readout_radarsplat_b7873200_fullscale.py",
        "RADARSPLAT_B7873200_PREPARE_CLEAN_STOP_RETAINED",
        "RADARSPLAT_B7873200_TARGET_CACHE_COMPLETE",
        "RADARSPLAT_B7873200_FULLSCALE_CLEAN_STOP_RETAINED",
        "RADARSPLAT_B7873200_FULLSCALE_PASS",
        "completion artifacts are present; no update executed.",
        "prepare_status[1]",
        "fit_status[1]",
        "fit_status[2]",
        "readout_status[1]",
        "terminal readout evidence exists",
    )
    for fragment in required_fragments:
        if fragment not in launcher:
            raise AssertionError(f"launcher is missing required fragment: {fragment}")
    for forbidden in ("--allow-development-subset", "--max-train 8", "--max-validation 4", "embers"):
        if forbidden in launcher:
            raise AssertionError(f"full-scale launcher contains an unapproved smoke/historical setting: {forbidden}")
    for forbidden in ("20000", "3000", "--steps 3200"):
        if re.search(rf"(?<![0-9]){forbidden}(?![0-9])", launcher):
            raise AssertionError(f"full-scale launcher contains an unapproved historical numeric setting: {forbidden}")

    readout = _source(ROOT / "scripts" / "readout_radarsplat_b7873200_fullscale.py")
    if "load_b787_power_arrays" in readout or "response_view" in readout:
        raise AssertionError("full-scale readout must remain cache-only")
    for fragment in (
        "reserved_test_materialized",
        "reserved_test_accessed",
        "zero_reference",
        "RADARSPLAT_B7873200_READOUT_RESOURCE_JSON=",
        "fit_lifecycle",
        "completion_resume",
        "readout_clean_interruption",
        "readout.log",
        "_phase_peak(completion_records",
        "_expected_validation_steps",
        "args.epochs",
        "args.sh_degree_interval != 600",
        "checkpoint_best.pt",
        "_select_best_validation_row",
        "minimum validation global native-power RelMSE",
        "earliest exact tie",
        "_active_sh_degree",
        "gaussian_occupancy_geometry_selected.npz",
        "export_gaussian_occupancy_geometry",
        "views_complete",
        "views_total",
        "same_checkpoint_pairing",
        "final_diagnostic",
        "relative_mse_formula",
        "B787_3200_CANONICAL_NPZ_PATH",
        "B787_3200_CANONICAL_MANIFEST_PATH",
        "target_provenance",
        "role_manifest_name",
        "train_view_ids",
        "validation_view_ids",
        "target_axes",
        "grid_crop",
        "matched_filter",
        "train_peak_power",
        "_learning_curve_budget_status",
        "384000",
        "480000",
        "running_best_validation_relative_mse_native_power",
        "automatic_extension",
        "descriptive_only",
    ):
        if fragment not in readout:
            raise AssertionError(f"full-scale readout lacks required selection/provenance marker: {fragment}")
    if "active_sh_degree=3" in readout:
        raise AssertionError("full-scale readout hardcodes terminal SH degree instead of deriving selected/final degree")
    if "sum_elevation(abs(matched_filter_complex)**2)" not in readout:
        raise AssertionError("full-scale readout does not state the native power observable")
    preparer = _source(ROOT / "scripts" / "prepare_radarsplat_b7873200_targets.py")
    for fragment in ("prepare_clean_interruption", "RADARSPLAT_B7873200_PREPARE_CLEAN_STOP_RETAINED", "_PREPARE_MATERIALIZED", "_process_peak_rss_kib() or 0"):
        if fragment not in preparer:
            raise AssertionError(f"full-scale preparation lacks explicit clean-stop support: {fragment}")
    _fixture_lifecycle_checks()
    _fixture_best_checkpoint_selection()
    _fixture_pipeline_return_codes()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate(args.config, args.launcher)
    print("RADARSPLAT_B7873200_FULLSCALE_CONFIG_PASS")


if __name__ == "__main__":
    main()
