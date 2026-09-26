#!/usr/bin/env python3
"""Torch-free source gate for the proposed full GeRaF B7873200 lane.

This check inspects only authored source text and syntax.  It does not open the
canonical archive, materialize a target, construct a model, submit a job, or
inspect a checkpoint.  The CUDA cache-transition regression and the existing
CPU contract remain separate runtime gates.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_NPZ = (
    "/storage/home/hcoda1/1/dbao31/r-jromberg3-0/RIFT/data/"
    "b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz"
)
CANONICAL_MANIFEST = (
    "/storage/scratch1/1/dbao31/rift_round8b_impl_20260810/splits/round8b/"
    "b78710k_interp_seed42_train3200_val1000_test1000_v1.json"
)
FULLSCALE_ROOT = "/storage/scratch1/1/dbao31/rift_b7873200_geraf_fullscale_v1"


class Gates:
    def __init__(self) -> None:
        self.count = 0

    def check(self, condition: bool, detail: str) -> None:
        if not condition:
            raise AssertionError(detail)
        self.count += 1
        print(f"PASS {self.count:02d}: {detail}", flush=True)


def _text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _tree(relative: str) -> ast.AST:
    path = ROOT / relative
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _has_function(tree: ast.AST, name: str) -> bool:
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
        for node in ast.walk(tree)
    )


def _has_default(source: str, option: str, value: str) -> bool:
    pattern = rf'add_argument\(\s*"{re.escape(option)}".{{0,300}}?default={re.escape(value)}'
    return re.search(pattern, source, flags=re.DOTALL) is not None


def main() -> None:
    gates = Gates()
    trainer = _text("train_geraf.py")
    trainer_tree = _tree("train_geraf.py")
    preparer = _text("scripts/prepare_geraf_b7873200_targets.py")
    preparer_tree = _tree("scripts/prepare_geraf_b7873200_targets.py")
    protocol = _text("rift/geraf_b7873200_protocol.py")
    source = _text("rift/geraf_b7873200_source.py")
    launcher = _text("slurm/train_geraf_b7873200_fullscale_v1.sbatch")

    required = (
        "train_geraf.py",
        "scripts/prepare_geraf_b7873200_targets.py",
        "rift/geraf_b7873200_protocol.py",
        "rift/geraf_b7873200_source.py",
        "rift/geraf_b7873200_preparation_state.py",
        "rift/geraf_b7873200_adapter.py",
        "rift/geraf_b7873200_acquisition.py",
        "rift/geraf_signal_operator.py",
        "rift/range_operator.py",
        "rift/serialized_range_operator.py",
        "scripts/validate_range_operator_inference_cache_transition.py",
        "scripts/validate_geraf_b7873200_protocol.py",
        "scripts/validate_geraf_b7873200_adapter.py",
        "scripts/validate_geraf_b7873200_acquisition.py",
        "scripts/validate_geraf_signal_operator.py",
        "scripts/validate_geraf_b7873200_trainer.py",
        "scripts/validate_geraf_b7873200_entrypoint.py",
        "scripts/validate_geraf_b7873200_preparation_state.py",
        "scripts/run_geraf_b78716_smallfit_timed.py",
    )
    gates.check(
        all((ROOT / relative).is_file() for relative in required),
        "the full-scale lane has an explicit source, cache, CUDA-regression, runtime-wrapper, and contract-validator closure",
    )
    gates.check(
        all(
            marker in protocol
            for marker in (
                "B787_3200_NUM_VIEWS = 10_000",
                "B787_3200_NUM_TRAIN = 3_200",
                "B787_3200_NUM_VALIDATION = 1_000",
                "B787_3200_NUM_TEST = 1_000",
                "B787_3200_NUM_UNUSED = 4_800",
            )
        )
        and "reserved_test" in source
        and "unused" in source,
        "the sealed protocol records the 10,000-view 3,200/1,000/1,000/4,800 partition and the source keeps reserved roles out of development access",
    )
    gates.check(
        "B787_3200_CANONICAL_NPZ_PATH" in trainer
        and "B787_3200_CANONICAL_MANIFEST_PATH" in trainer
        and "B787_3200_CANONICAL_NPZ_PATH" in preparer
        and "B787_3200_CANONICAL_MANIFEST_PATH" in preparer
        and "b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz" in protocol
        and "b78710k_interp_seed42_train3200_val1000_test1000_v1.json" in protocol,
        "trainer and preparer bind the exact canonical sphere10k archive and frozen interpolation manifest",
    )
    gates.check(
        all(
            _has_default(trainer, option, value)
            for option, value in (
                ("--steps", "50_000"),
                ("--n-azimuth", "32"),
                ("--n-elevation", "32"),
                ("--n-depth", "32"),
                ("--phase-sign", "-1.0"),
                ("--compute-dtype", '"float64"'),
                ("--oversample", "2"),
                ("--kernel-width", "20"),
                ("--pair-chunk", "32"),
                ("--point-chunk", "4096"),
                ("--validation-every", "1000"),
            )
        ),
        "the trainer defaults preserve the explicit full-scale 32^3, phase-minus-one, float64, range-NUFFT, 50,000-update recipe",
    )
    gates.check(
        all(
            _has_default(preparer, option, value)
            for option, value in (
                ("--n-azimuth", "32"),
                ("--n-elevation", "32"),
                ("--n-depth", "32"),
                ("--phase-sign", "-1.0"),
                ("--compute-dtype", '"float64"'),
                ("--oversample", "2"),
                ("--kernel-width", "20"),
                ("--pair-chunk", "32"),
                ("--point-chunk", "4096"),
                ("--freq-chunk", "75"),
                ("--max-views", "0"),
            )
        )
        and "selected = list(authorized)" in preparer
        and "response_roles_materialized" in preparer,
        "the preparer defaults to all 4,200 train/validation targets and retains its smoke cap as an explicit non-production opt-in",
    )
    gates.check(
        "load_b7873200_metadata_source" in trainer
        and "validate_b7873200_target_cache" in trainer
        and "response_payload_materialized" in trainer
        and "response_view" in source,
        "training binds a metadata-only source and a complete cache before model construction; raw response access remains preparation-only",
    )
    gates.check(
        _has_function(trainer_tree, "_publish_timed_wrapper_ready")
        and "CLEAN_STOP_EXIT_CODE = 143" in trainer
        and "signal.signal(signal.SIGTERM, _request_stop)" in trainer
        and "signal.signal(signal.SIGINT, _request_stop)" in trainer
        and "_publish_timed_wrapper_ready()" in trainer,
        "the full trainer publishes wrapper readiness only after its cooperative TERM/INT handlers are installed",
    )
    gates.check(
        _has_function(preparer_tree, "_prepare_one")
        and "PREPARATION_STATE_FILENAME" in preparer
        and "_write_preparation_state" in preparer
        and "complete_preparation_phase" in preparer
        and "completion_exit" in preparer
        and "signal.signal(signal.SIGTERM, _request_stop)" in preparer
        and "finish the current target" in preparer
        and "with torch.inference_mode()" in preparer
        and "matched_filter_complex" in preparer
        and "fit_split" in preparer
        and '"clip": False' in preparer,
        "target construction uses train-only normalization and inference-mode native matched-filter preparation",
    )
    gates.check(
        CANONICAL_NPZ in launcher
        and CANONICAL_MANIFEST in launcher
        and FULLSCALE_ROOT in launcher
        and "--max-views" not in launcher
        and "--steps 50000" in launcher
        and "--validation-every 1000" in launcher
        and "--compute-dtype float64" in launcher,
        "the proposed launcher cannot silently become a smoke run and names fresh full-scale cache/fit roots",
    )
    gates.check(
        "scripts/validate_range_operator_inference_cache_transition.py" in launcher
        and "scripts/prepare_geraf_b7873200_targets.py" in launcher
        and "train_geraf.py" in launcher
        and "scripts/run_geraf_b78716_smallfit_timed.py" in launcher
        and "checkpoint_latest.pth.tar" in launcher
        and "checkpoint_final.pth.tar" in launcher,
        "one allocation orders the repaired cache regression, complete target preparation, timed fit, and terminal checkpoint/readout checks",
    )
    lifecycle = (
        "GERAF_B7873200_FULLSCALE_SOURCE_PREFLIGHT_START",
        "GERAF_B7873200_FULLSCALE_CACHE_TRANSITION_START",
        "GERAF_B7873200_FULLSCALE_PREPARE_START",
        "GERAF_B7873200_FULLSCALE_FIT_START",
        "GERAF_B7873200_FULLSCALE_READOUT_START",
    )
    offsets = [launcher.index(marker) for marker in lifecycle]
    gates.check(
        offsets == sorted(offsets)
        and "run_mode=fresh" in launcher
        and "run_mode=resume" in launcher
        and "--resume" in launcher,
        "the candidate launcher has explicit fresh/resume dispatch around one ordered preparation/fit/readout lifecycle; phase-aware manager wiring remains a release TODO",
    )
    gates.check(
        "#SBATCH --account=gts-jromberg3-ece" in launcher
        and "#SBATCH --qos=inferno" in launcher
        and "#SBATCH --time=12:00:00" in launcher
        and 'SLURM_JOB_QOS:-}" == "inferno"' in launcher
        and 'SLURM_JOB_ACCOUNT:-}" == "gts-jromberg3-ece"' in launcher
        and "#SBATCH --partition=gpu-h100" in launcher
        and "#SBATCH --gres=gpu:h100:1" in launcher
        and "#SBATCH --signal=TERM@600" in launcher
        and "PREP_TIME_LOG" in launcher
        and "FIT_ALLOCATION_TIME_LOG" in launcher
        and "FIT_PROCESS_TIME_LOG" in launcher
        and "READOUT_TIME_LOG" in launcher
        and "existing_time_logs(\"preparation\")" in launcher
        and "existing_time_logs(\"fit_allocation\")" in launcher
        and "existing_time_logs(\"fit_process\")" in launcher
        and "phase_time_logs[\"readout\"]" in launcher,
        "the candidate resource envelope and retained per-phase GNU-time records cover preparation, fit, and readout together",
    )
    print(f"GERAF_B7873200_FULLSCALE_SOURCE_PASS gates={gates.count}", flush=True)


if __name__ == "__main__":
    main()
