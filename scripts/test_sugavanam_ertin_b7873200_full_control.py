#!/usr/bin/env python3
"""Small Torch-free control tests for the full-package recovery paths."""

from __future__ import annotations

import ast
import argparse
from contextlib import contextmanager
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
from typing import Mapping
import uuid


ROOT = Path(__file__).resolve().parents[1]
DRIVER = (ROOT / "train_sugavanam_ertin.py").read_text(encoding="utf-8")
STAGE2_DRIVER = (ROOT / "train_sugavanam_ertin_stage2.py").read_text(encoding="utf-8")
LAUNCHER = (ROOT / "slurm/validate_sugavanam_ertin_b7873200_full_v1.sbatch").read_text(encoding="utf-8")
TEST_TMP_PARENT = Path(os.environ.get("SE_TEST_TMP_PARENT", str(ROOT)))
BASH_FIXTURE_SKIP_COUNT = 0


@contextmanager
def _test_directory(prefix: str):
    """Use a direct workspace directory; Windows tempfile ACLs reject nested mkdirs."""

    TEST_TMP_PARENT.mkdir(parents=True, exist_ok=True)
    # Keep the generated leaf short: the synthetic run/checkpoint identity is
    # intentionally long and Windows path limits otherwise mask the controls.
    directory = TEST_TMP_PARENT / f"se_{uuid.uuid4().hex}"
    directory.mkdir()
    try:
        yield str(directory)
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _skip_bash_fixture_on_windows(name: str) -> bool:
    global BASH_FIXTURE_SKIP_COUNT
    if os.name == "nt":
        BASH_FIXTURE_SKIP_COUNT += 1
        print(f"SE_B7873200_FULL_BASH_FIXTURE_SKIP_WINDOWS: {name}")
        return True
    return False


def _assert_order(first: str, second: str, label: str) -> None:
    left = DRIVER.find(first)
    right = DRIVER.find(second)
    assert left >= 0 and right >= 0 and left < right, f"{label}: expected ordered source path"


def _mock_stage2_outcome(returned: int, lifecycle_clean: bool) -> str:
    """Model the driver policy without importing Torch or the driver."""

    if returned == 143 and lifecycle_clean:
        return "clean_interruption"
    return "technical_execution_failure"


def test_returned_143_is_not_a_failure() -> None:
    assert _mock_stage2_outcome(143, True) == "clean_interruption"
    assert _mock_stage2_outcome(143, False) == "technical_execution_failure"
    assert _mock_stage2_outcome(1, True) == "technical_execution_failure"
    _assert_order("if result == CLEAN_INTERRUPTION_EXIT_CODE:", "if result != 0:", "returned 143")
    assert "_validate_stage2_returned_clean(stage2_contract, provenance, recipe)" in DRIVER


def test_raised_system_exit_is_distinct() -> None:
    tree = ast.parse(DRIVER)
    handlers = [node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler)]
    assert sum(1 for node in handlers if isinstance(node.type, ast.Name) and node.type.id == "SystemExit") >= 2
    assert "code = exc.code if isinstance(exc.code, int) else 1" in DRIVER
    assert "if code == CLEAN_INTERRUPTION_EXIT_CODE:" in DRIVER


def test_clean_report_and_checkpoint_evidence_are_validated() -> None:
    assert "previous.get(\"status\") != \"clean_interruption\"" in DRIVER
    assert "_validate_stage1_clean_record(paths)" in DRIVER
    assert "_validate_stage2_clean_record(paths)" in DRIVER
    assert "checkpoint_path" in DRIVER and "resume_allowed" in DRIVER


def test_compute_cap_report_is_nonterminal_signal_only_and_stage2_free() -> None:
    assert '"mode": "stage1_compute_capped"' in DRIVER
    assert '"underlying_recipe_epochs": 150' in DRIVER
    assert '"total_epoch_budget": FULL_STAGE1_COMPUTE_CAP_EPOCH' in DRIVER
    assert '"geometry_metrics": "N/A"' in DRIVER
    assert '"geometry_status": "not_applicable_no_stage2_compute_cap"' in DRIVER
    assert '"auto_resume_normal_full_pipeline": False' in DRIVER
    assert '"status": "not_run_compute_cap"' in DRIVER
    assert 'status == "compute_capped_stage1"' in DRIVER
    _assert_order(
        'print(f"SE_B7873200_FULL_STAGE1_COMPUTE_CAPPED_EPOCH30:',
        'return 0',
        "compute-cap completion marker",
    )


def test_term_at_epoch30_prefers_terminal_cap_boundary() -> None:
    helpers = _load_resume_helpers()
    decide = helpers["_compute_cap_stop_reason"]
    assert decide(epoch=30, cap_epoch=30, signal_requested=True) == "compute_cap"
    assert decide(epoch=29, cap_epoch=30, signal_requested=True) == "signal"
    assert "Stage1EngineeringObserver.on_epoch_end" in DRIVER
    assert DRIVER.index("reason = _compute_cap_stop_reason(") < DRIVER.index(
        'if reason == "compute_cap":'
    )


def test_stage1_clean_latest_resume_requires_companion_record() -> None:
    helpers = _load_resume_helpers()
    with _test_directory("se_b7873200_stage1_clean_resume_") as temporary:
        parent = Path(temporary)
        run_root = parent / "b78710k_sugavanam_ertin_full3200_stage1_stage2_replacement_v1"
        generic_root = run_root / "stage1/sugavanam_ertin_b7873200_stage1_full3200_replacement_v1"
        latest = generic_root / "checkpoint_latest.pth.tar"
        clean = generic_root / "full_stage1_clean_interruption.json"
        report = run_root / "full3200_report.json"
        generic_root.mkdir(parents=True)
        latest.write_bytes(b"latest")
        clean.write_text(json.dumps({
            "schema": "rift_sugavanam_ertin_b7873200_full_stage1_recovery_v1",
            "phase": "stage1",
            "state": "clean_interrupted",
            "resume_allowed": True,
            "checkpoint_path": str(latest),
            "checkpoint_role": "latest",
        }), encoding="utf-8")
        report.write_text(json.dumps({"status": "clean_interruption", "phase": "stage1"}), encoding="utf-8")
        args = argparse.Namespace(
            checkpoint_root=str(parent),
            npz_path=helpers["B787_3200_CANONICAL_NPZ_PATH"],
            parent_role_manifest=helpers["B787_3200_CANONICAL_MANIFEST_PATH"],
            device="cuda",
            resume=str(latest),
        )
        paths = helpers["_paths"](args)
        helpers["_validate_identity"](args, paths)

        clean.unlink()
        try:
            helpers["_validate_identity"](args, paths)
        except helpers["FullContractError"]:
            pass
        else:
            raise AssertionError("Stage-1 latest resume accepted a report without a clean record")

        clean.write_text(json.dumps({
            "state": "clean_interrupted",
            "resume_allowed": True,
            "checkpoint_path": str(latest),
        }), encoding="utf-8")
        latest.unlink()
        try:
            helpers["_validate_identity"](args, paths)
        except helpers["FullContractError"]:
            pass
        else:
            raise AssertionError("Stage-1 latest resume accepted a missing checkpoint")

        latest.write_bytes(b"latest")
        (generic_root / "checkpoint_final.pth.tar").write_bytes(b"final")
        try:
            helpers["_validate_identity"](args, paths)
        except helpers["FullContractError"]:
            pass
        else:
            raise AssertionError("Stage-1 latest resume accepted a terminal final artifact")


def test_compute_cap_resume_is_total_epoch_30_and_requires_clean_source() -> None:
    helpers = _load_resume_helpers()
    checkpoint_epochs: dict[str, int] = {}
    helpers["torch"] = SimpleNamespace(
        load=lambda path, **kwargs: {"epoch": checkpoint_epochs[str(path)]}
    )
    with _test_directory("se_b7873200_compute_cap_") as temporary:
        parent = Path(temporary)
        source = (
            parent / "b78710k_sugavanam_ertin_full3200_stage1_stage2_replacement_v1"
            / "stage1/sugavanam_ertin_b7873200_stage1_full3200_replacement_v1/checkpoint_latest.pth.tar"
        )
        source.parent.mkdir(parents=True)
        source.write_bytes(b"source")
        checkpoint_epochs[str(source)] = 7
        clean = source.parent / "full_stage1_clean_interruption.json"
        clean.write_text(json.dumps({
            "state": "clean_interrupted",
            "resume_allowed": True,
            "checkpoint_path": str(source),
        }), encoding="utf-8")
        args = argparse.Namespace(
            checkpoint_root=str(parent),
            npz_path=helpers["B787_3200_CANONICAL_NPZ_PATH"],
            parent_role_manifest=helpers["B787_3200_CANONICAL_MANIFEST_PATH"],
            device="cuda",
            resume=str(source),
            compute_capped_stage1_epoch30=True,
            compute_capped_stage1_source_checkpoint=str(source),
        )
        paths = helpers["_paths"](args)
        helpers["_validate_identity"](args, paths)
        assert paths["compute_cap"] is True
        assert paths["compute_cap_source"]["origin_epoch"] == 7
        assert paths["compute_cap_source"]["source_epoch"] == 7
        assert paths["compute_cap_epoch"] == 30

        clean.unlink()
        try:
            helpers["_validate_identity"](args, paths)
        except helpers["FullContractError"]:
            pass
        else:
            raise AssertionError("compute cap accepted a missing source clean record")

        clean.write_text(json.dumps({
            "state": "clean_interrupted",
            "resume_allowed": True,
            "checkpoint_path": str(source),
        }), encoding="utf-8")
        checkpoint_epochs[str(source)] = 30
        try:
            helpers["_validate_identity"](args, paths)
        except helpers["FullContractError"]:
            pass
        else:
            raise AssertionError("compute cap accepted a source already at the target epoch")

        # A later 12-hour chunk resumes the cap root, while preserving the
        # original epoch-7 provenance rather than relabeling epoch 17 as the start.
        checkpoint_epochs[str(source)] = 7
        clean.write_text(json.dumps({
            "state": "clean_interrupted",
            "resume_allowed": True,
            "checkpoint_path": str(source),
        }), encoding="utf-8")
        cap_latest = (
            parent / "b78710k_sugavanam_ertin_stage1_compute_capped_epoch30_v1"
            / "stage1/sugavanam_ertin_b7873200_stage1_compute_capped_epoch30_v1/checkpoint_latest.pth.tar"
        )
        cap_latest.parent.mkdir(parents=True)
        cap_latest.write_bytes(b"cap")
        checkpoint_epochs[str(cap_latest)] = 17
        cap_clean = cap_latest.parent / "full_stage1_clean_interruption.json"
        cap_clean.write_text(json.dumps({
            "state": "clean_interrupted",
            "resume_allowed": True,
            "checkpoint_path": str(cap_latest),
            "origin_checkpoint_path": str(source),
            "origin_epoch": 7,
            "origin_resource_reference": str(source.parents[2] / "resource_accounting.json"),
        }), encoding="utf-8")
        (cap_latest.parents[2] / "stage1_compute_capped_epoch30_report.json").write_text(
            json.dumps({"status": "clean_interruption", "phase": "stage1_compute_capped"}),
            encoding="utf-8",
        )
        args.resume = str(cap_latest)
        args.compute_capped_stage1_source_checkpoint = str(cap_latest)
        continued = helpers["_paths"](args)
        helpers["_validate_identity"](args, continued)
        assert continued["compute_cap_source"]["source_epoch"] == 17
        assert continued["compute_cap_source"]["origin_epoch"] == 7


def test_compute_cap_boundary_rejects_final_bundle_and_stage2() -> None:
    helpers = _load_resume_helpers()
    checkpoint_epochs: dict[str, int] = {}
    helpers["torch"] = SimpleNamespace(
        load=lambda path, **kwargs: {"epoch": checkpoint_epochs[str(path)]}
    )
    with _test_directory("se_b7873200_compute_cap_boundary_") as temporary:
        parent = Path(temporary)
        args = argparse.Namespace(
            checkpoint_root=str(parent),
            npz_path=helpers["B787_3200_CANONICAL_NPZ_PATH"],
            parent_role_manifest=helpers["B787_3200_CANONICAL_MANIFEST_PATH"],
            device="cuda",
            resume=None,
            compute_capped_stage1_epoch30=True,
            compute_capped_stage1_source_checkpoint=None,
        )
        paths = helpers["_paths"](args)
        paths["root"].mkdir(parents=True)
        paths["stage1_generic"].mkdir(parents=True)
        paths["stage1_latest"].write_bytes(b"latest")
        checkpoint_epochs[str(paths["stage1_latest"])] = 30
        paths["stage1_clean"].write_text(json.dumps({
            "state": "clean_interrupted",
            "resume_allowed": False,
            "terminal": True,
            "target_epoch": 30,
            "checkpoint_path": str(paths["stage1_latest"]),
        }), encoding="utf-8")
        helpers["_validate_compute_cap_boundary"](paths)
        paths["stage1_final"].write_bytes(b"final")
        try:
            helpers["_validate_compute_cap_boundary"](paths)
        except helpers["FullContractError"]:
            pass
        else:
            raise AssertionError("compute cap boundary accepted checkpoint_final")


def test_terminal_final_recovery_is_zero_update() -> None:
    assert "_stage1_argv(paths, paths[\"stage1_final\"], resume_path=str(paths[\"stage1_final\"]))" in DRIVER
    assert "recovery_without_optimizer_updates" in DRIVER
    assert "native.get(\"observed_optimizer_steps\") != 0" in DRIVER
    assert "terminal_final_readout_recovery_without_optimizer_updates" in DRIVER


def test_original_metrics_and_run_wide_resources_are_persisted() -> None:
    assert "original_first_attempt" in DRIVER
    assert "preserved_first_attempt" in DRIVER
    assert "full_stage1_native_readouts.json" in DRIVER
    assert "attempts" in DRIVER and "cumulative_work_wall_seconds" in DRIVER
    assert "cuda_peak_reserved_bytes" in DRIVER


def test_signal_forwarding_reaches_cooperative_handlers() -> None:
    assert "trap forward_term TERM INT" in LAUNCHER
    assert 'kill -TERM "$driver_pid"' in LAUNCHER
    assert "install_stop_handlers()" in DRIVER
    assert "if stop_requested():" in DRIVER
    assert "raise SystemExit(CLEAN_INTERRUPTION_EXIT_CODE)" in DRIVER
    assert "mkfifo -- \"$driver_fifo\"" in LAUNCHER
    assert "tee_pid=$!" in LAUNCHER
    assert "term_requested=0" in LAUNCHER
    assert "term_forwarded=0" in LAUNCHER
    assert "SE_B7873200_FULL_TERM_BEFORE_DRIVER_EXIT" in LAUNCHER
    assert "SE_B7873200_FULL_TERM_DURING_TEE_DRAIN" in LAUNCHER
    assert "SE_B7873200_FULL_REAP_AFTER_INTERRUPTED_TEE_WAIT" in LAUNCHER
    assert "SE_B7873200_FULL_WAITING_FOR_DRIVER_CLEAN_STOP" not in LAUNCHER
    assert 'wait "$driver_pid"' in LAUNCHER
    _assert_launcher_order(
        'driver_pid=\'\'\nwait "$tee_pid"',
        "SE_B7873200_FULL_REAP_AFTER_INTERRUPTED_TEE_WAIT",
        "tee drain after driver reap",
    )
    assert "SE_B7873200_FULL_CLEAN_INTERRUPTION_RECORDED" in LAUNCHER
    assert "preserve_existing_stop=True" in DRIVER
    assert "_record_stage2_resume_clean" in DRIVER
    assert "stage2_resume=stage2_resume" in DRIVER
    assert 'STAGE1_CHECKPOINT_NAME=sugavanam_ertin_b7873200_stage1_full3200_replacement_v1' in LAUNCHER
    assert 'STAGE1_LATEST="$STAGE1_DIR/checkpoint_latest.pth.tar"' in LAUNCHER
    assert 'python - "$REPORT" "$STAGE1_DIR" "$STAGE1_BUNDLE"' in LAUNCHER
    assert "stage1-compute-cap30" in LAUNCHER
    assert "CAP_SOURCE_LATEST" in LAUNCHER
    assert "--compute-capped-stage1-epoch30" in LAUNCHER
    _assert_launcher_order(
        'elif [[ -f "$STAGE1_BUNDLE" ]]',
        'elif [[ -f "$STAGE1_FINAL" ]]',
        "semantic bundle resume priority",
    )


def test_launcher_rejects_lone_stage1_latest() -> None:
    assert "stage1_latest_resume_ok()" in LAUNCHER
    assert 'local latest="${1:-$STAGE1_LATEST}"' in LAUNCHER
    assert 'local clean="${2:-$STAGE1_CLEAN}"' in LAUNCHER
    assert "! stage1_latest_resume_ok &&" in LAUNCHER
    assert 'elif stage1_latest_resume_ok; then' in LAUNCHER
    _assert_launcher_order(
        '! stage1_latest_resume_ok &&',
        'elif [[ -f "$STAGE1_FINAL" ]]',
        "latest clean-record gate before final fallback",
    )


def _assert_launcher_order(first: str, second: str, label: str) -> None:
    left = LAUNCHER.find(first)
    right = LAUNCHER.find(second)
    assert left >= 0 and right >= 0 and left < right, f"{label}: expected ordered launcher path"


def test_resource_postflight_validation_is_executable() -> None:
    postflight_path = ROOT / "scripts/postflight_sugavanam_ertin_b7873200_full.py"
    spec = importlib.util.spec_from_file_location("full_postflight_control", postflight_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    resource = {
        "schema": "rift_sugavanam_ertin_b7873200_full_resource_accounting_v2",
        "per_attempt_wall_limit_seconds": 8 * 3600,
        "attempts": [{
            "attempt_index": 1,
            "total_wall_seconds": 1.0,
            "phases": {
                "stage1": {
                    "execution": "training",
                    "wall_seconds": 0.4,
                    "process_peak_rss_bytes": 1.0,
                    "cuda_peak_allocated_bytes": 0.0,
                    "cuda_peak_reserved_bytes": 0.0,
                },
                "stage2": {
                    "execution": "stage2_lifecycle",
                    "wall_seconds": 0.6,
                    "process_peak_rss_bytes": 1.0,
                    "cuda_peak_allocated_bytes": 0.0,
                    "cuda_peak_reserved_bytes": 0.0,
                },
            },
        }],
        "cumulative_work_wall_seconds": 1.0,
    }
    checked = module._check_resource(resource, host_limit_gib=64.0, time_limit_hours=8.0)
    assert checked["attempt_count"] == 1
    broken = copy.deepcopy(resource)
    broken["cumulative_work_wall_seconds"] = 0.0
    try:
        module._check_resource(broken, host_limit_gib=64.0, time_limit_hours=8.0)
    except ValueError as exc:
        assert "cumulative" in str(exc)
    else:
        raise AssertionError("postflight accepted an inconsistent cumulative resource ledger")


def _load_resume_helpers() -> dict[str, object]:
    tree = ast.parse(DRIVER)
    wanted = {
        "_compute_cap_stop_reason",
        "_same_path",
        "_paths",
        "_validate_stage1_clean_record",
        "_validate_compute_cap_source",
        "_validate_compute_cap_boundary",
        "_validate_stage2_clean_record",
        "_validate_stage2_entry_clean_record",
        "_record_stage2_resume_clean",
        "_validate_identity",
    }
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    namespace: dict[str, object] = {
        "Path": Path,
        "Mapping": Mapping,
        "os": os,
        "json": json,
        "argparse": argparse,
        "FULL_RUN_NAME": "b78710k_sugavanam_ertin_full3200_stage1_stage2_replacement_v1",
        "FULL_REPORT_NAME": "full3200_report.json",
        "FULL_STAGE1_CHECKPOINT_NAME": "sugavanam_ertin_b7873200_stage1_full3200_replacement_v1",
        "FULL_STAGE1_BUNDLE_FILENAME": "sugavanam_ertin_b7873200_stage1_final_v1.pth.tar",
        "FULL_STAGE1_COMPUTE_CAP_EPOCH": 30,
        "FULL_STAGE1_COMPUTE_CAP_RUN_NAME": "b78710k_sugavanam_ertin_stage1_compute_capped_epoch30_v1",
        "FULL_STAGE1_COMPUTE_CAP_CHECKPOINT_NAME": "sugavanam_ertin_b7873200_stage1_compute_capped_epoch30_v1",
        "FULL_STAGE1_COMPUTE_CAP_REPORT_NAME": "stage1_compute_capped_epoch30_report.json",
        "FULL_STAGE1_COMPUTE_CAP_RESOURCE_NAME": "stage1_compute_capped_epoch30_resource_accounting.json",
        "B787_3200_CANONICAL_NPZ_PATH": "/storage/home/hcoda1/1/dbao31/r-jromberg3-0/RIFT/data/b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz",
        "B787_3200_CANONICAL_MANIFEST_PATH": "/storage/scratch1/1/dbao31/rift_round8b_impl_20260810/splits/round8b/b78710k_interp_seed42_train3200_val1000_test1000_v1.json",
        "FullContractError": type("FullContractError", (ValueError,), {}),
        "CLEAN_INTERRUPTION_EXIT_CODE": 143,
        "load_json_mapping": lambda path, label: json.loads(Path(path).read_text(encoding="utf-8")),
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), "full_resume_helpers", "exec"), namespace)
    return namespace


def test_resume_helper_accepts_bundle_before_final_fallback() -> None:
    helpers = _load_resume_helpers()
    with _test_directory("se_b7873200_resume_") as temporary:
        parent = Path(temporary)
        run_root = parent / "b78710k_sugavanam_ertin_full3200_stage1_stage2_replacement_v1"
        stage1_root = run_root / "stage1"
        generic_root = stage1_root / "sugavanam_ertin_b7873200_stage1_full3200_replacement_v1"
        bundle = stage1_root / "sugavanam_ertin_b7873200_stage1_final_v1.pth.tar"
        final = generic_root / "checkpoint_final.pth.tar"
        generic_root.mkdir(parents=True)
        bundle.write_bytes(b"bundle")
        final.write_bytes(b"final")
        args = argparse.Namespace(
            checkpoint_root=str(parent),
            npz_path=helpers["B787_3200_CANONICAL_NPZ_PATH"],
            parent_role_manifest=helpers["B787_3200_CANONICAL_MANIFEST_PATH"],
            device="cuda",
            resume=str(bundle),
        )
        paths = helpers["_paths"](args)
        helpers["_validate_identity"](args, paths)
        bundle.unlink()
        args.resume = str(final)
        helpers["_validate_identity"](args, paths)


def test_repeated_stage2_resume_stop_preserves_lifecycle_latest() -> None:
    helpers = _load_resume_helpers()
    with _test_directory("se_b7873200_stage2_resume_") as temporary:
        parent = Path(temporary)
        run_root = parent / "b78710k_sugavanam_ertin_full3200_stage1_stage2_replacement_v1"
        stage2_root = run_root / "stage2"
        latest = stage2_root / "checkpoint_latest.pth.tar"
        lifecycle = stage2_root / "lifecycle.json"
        stage2_root.mkdir(parents=True)
        latest.write_bytes(b"latest")
        lifecycle.write_text(json.dumps({
            "state": "clean_interrupted",
            "resume_allowed": True,
            "checkpoint_path": str(latest),
        }), encoding="utf-8")
        args = argparse.Namespace(
            checkpoint_root=str(parent),
            npz_path=helpers["B787_3200_CANONICAL_NPZ_PATH"],
            parent_role_manifest=helpers["B787_3200_CANONICAL_MANIFEST_PATH"],
            device="cuda",
            resume=str(latest),
        )
        paths = helpers["_paths"](args)
        writes: list[dict[str, object]] = []
        reports: list[dict[str, object]] = []
        helpers["_write_resource"] = lambda *args, **kwargs: writes.append(kwargs)
        helpers["_report"] = lambda *args, **kwargs: reports.append(kwargs)
        for _ in range(2):
            result = helpers["_record_stage2_resume_clean"](
                paths,
                started=0.0,
                normalization=None,
                native_metrics=None,
                stage1_recipe=None,
                stage1_resource={"execution": "training", "wall_seconds": 0.1},
                stage2_recipe={"steps": 5_000},
            )
            assert result == 143
        assert len(writes) == 2 and all(item["phase"] == "stage2" for item in writes)
        assert len(reports) == 2 and all(item["phase"] == "stage2" for item in reports)
        assert not paths["stage2_entry_clean"].exists()
        assert latest.is_file() and lifecycle.is_file()


def _load_stage2_run_seam() -> tuple[dict[str, object], dict[str, int | bool]]:
    tree = ast.parse(STAGE2_DRIVER)
    run_node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_run")
    state: dict[str, int | bool] = {"stop": True, "reset": 0, "install": 0}

    class _DefaultReached(RuntimeError):
        pass

    class _FakeTorch:
        @staticmethod
        def device(value: str) -> str:
            return value

    def validate_args(*args: object, **kwargs: object) -> None:
        del args, kwargs

    def lifecycle_contract_record(value: Mapping[str, object]) -> dict[str, object]:
        del value
        return {}

    def provenance_output_identity(value: Mapping[str, object]) -> dict[str, object]:
        del value
        return {}

    def prepare_run_root(*args: object, **kwargs: object) -> str:
        assert kwargs["resume"] == "stage2-latest"
        return "resume"

    def stop_requested() -> bool:
        return bool(state["stop"])

    def reset_stop_request() -> None:
        state["reset"] = int(state["reset"]) + 1
        state["stop"] = False

    def install_stop_handlers() -> None:
        state["install"] = int(state["install"]) + 1

    def set_seed(value: int) -> None:
        del value
        raise _DefaultReached("default reset path reached the old initialization boundary")

    namespace: dict[str, object] = {
        "copy": copy,
        "Mapping": Mapping,
        "torch": _FakeTorch(),
        "validate_args": validate_args,
        "lifecycle_contract_record": lifecycle_contract_record,
        "provenance_output_identity": provenance_output_identity,
        "prepare_run_root": prepare_run_root,
        "stop_requested": stop_requested,
        "reset_stop_request": reset_stop_request,
        "install_stop_handlers": install_stop_handlers,
        "_set_seed": set_seed,
        "CLEAN_INTERRUPTION_EXIT_CODE": 143,
        "Stage2ContractError": type("Stage2ContractError", (ValueError,), {}),
    }
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations", asname=None)], level=0
    )
    seam_module = ast.fix_missing_locations(ast.Module(body=[future, run_node], type_ignores=[]))
    exec(compile(seam_module, "stage2_run_seam", "exec"), namespace)
    return namespace, state


def test_stage2_stop_seam_preserves_existing_flag_and_default_resets() -> None:
    namespace, state = _load_stage2_run_seam()
    contract = SimpleNamespace(
        output_dir="stage2",
        latest_checkpoint="stage2/checkpoint_latest.pth.tar",
        complete_dir="stage2/complete",
        lifecycle_path="stage2/lifecycle.json",
    )
    args = SimpleNamespace(device="cuda", resume="stage2-latest")
    common = {
        "contract": contract,
        "recipe": {"steps": 1, "seed": 1},
        "source": {"validated": True},
        "provenance": {"validated": True},
        "engineering_override": True,
    }
    try:
        namespace["_run"](args, **common, preserve_existing_stop=True)
    except SystemExit as exc:
        assert exc.code == 143
    else:
        raise AssertionError("preserved Stage-2 stop state was cleared")
    assert state["reset"] == 0 and state["install"] == 0
    state["stop"] = True
    try:
        namespace["_run"](args, **common)
    except RuntimeError as exc:
        assert "old initialization boundary" in str(exc)
    else:
        raise AssertionError("standalone Stage-2 default did not reach its reset path")
    assert state["reset"] == 1 and state["install"] == 1 and state["stop"] is False


def _resolve_bash() -> Path:
    if os.name == "nt":
        known_git_bash = Path(r"C:\Users\dbao31\AppData\Local\Programs\Git\bin\bash.exe")
        if known_git_bash.is_file():
            return known_git_bash
    else:
        system_bash = Path("/bin/bash")
        if system_bash.is_file():
            return system_bash
    discovered = shutil.which("bash")
    assert discovered, "a Bash executable is required for the portable launcher fixture"
    return Path(discovered)


def test_bash_wait_fixture_reaps_after_interrupted_wait() -> None:
    if _skip_bash_fixture_on_windows("driver_wait_reap"):
        return
    git_bash = _resolve_bash()
    fixture = r'''set -euo pipefail
fixture_dir="$(mktemp -d)"
trap 'rm -rf -- "$fixture_dir"' EXIT
clean_fifo="$fixture_dir/clean.fifo"
clean_log="$fixture_dir/clean.log"
mkfifo -- "$clean_fifo"
tee "$clean_log" < "$clean_fifo" &
tee_pid=$!
driver_pid=''
forward_term() {
  term_requested=1
  if [[ -n "$driver_pid" ]]; then
    if (( term_forwarded == 0 )); then
      term_forwarded=1
      kill -TERM -- "-$driver_pid" 2>/dev/null || kill -TERM "$driver_pid" 2>/dev/null || true
    fi
  fi
}
term_requested=0
term_forwarded=0
trap forward_term TERM INT
if command -v setsid >/dev/null 2>&1; then
  setsid bash -c 'trap "echo driver-clean-stop; exit 0" TERM; sleep 5' > "$clean_fifo" 2>&1 &
else
  bash -c 'trap "echo driver-clean-stop; exit 0" TERM; sleep 5' > "$clean_fifo" 2>&1 &
fi
driver_pid=$!
(sleep 0.2; kill -TERM "$$") &
set +e
wait "$driver_pid"
first_wait_status=$?
if (( term_requested && (first_wait_status == 143 || first_wait_status == 130) )); then
  wait "$driver_pid"
  reap_status=$?
  driver_status="$reap_status"
else
  driver_status="$first_wait_status"
fi
wait "$tee_pid"
tee_status=$?
set -e
trap - TERM INT
driver_pid=''
grep -Fq driver-clean-stop "$clean_log"

failure_fifo="$fixture_dir/failure.fifo"
failure_log="$fixture_dir/failure.log"
mkfifo -- "$failure_fifo"
tee "$failure_log" < "$failure_fifo" &
failure_tee_pid=$!
bash -c 'echo driver-failure; exit 7' > "$failure_fifo" 2>&1 &
failure_driver_pid=$!
set +e
wait "$failure_driver_pid"
failure_driver_status=$?
wait "$failure_tee_pid"
failure_tee_status=$?
set -e
grep -Fq driver-failure "$failure_log"
printf 'clean_first_wait=%s clean_driver_status=%s clean_tee_status=%s failure_driver_status=%s failure_tee_status=%s\n' "$first_wait_status" "$driver_status" "$tee_status" "$failure_driver_status" "$failure_tee_status"
[[ "$first_wait_status" == 143 ]]
[[ "$driver_status" == 0 ]]'''
    result = subprocess.run(
        [str(git_bash), "-lc", fixture],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "clean_first_wait=143 clean_driver_status=0 clean_tee_status=0 failure_driver_status=7 failure_tee_status=0" in result.stdout


def test_bash_term_before_driver_exits_without_wait_spin() -> None:
    if _skip_bash_fixture_on_windows("term_before_driver"):
        return
    git_bash = _resolve_bash()
    fixture = r'''set -euo pipefail
driver_pid=''
term_requested=0
forward_term() {
  term_requested=1
  if [[ -n "$driver_pid" ]]; then
    kill -TERM -- "-$driver_pid" 2>/dev/null || kill -TERM "$driver_pid" 2>/dev/null || true
  else
    echo TERM_BEFORE_DRIVER_START
  fi
}
trap forward_term TERM INT
(sleep 0.05; kill -TERM "$$") &
sleep 0.2
if (( term_requested )); then
  trap - TERM INT
  echo TERM_BEFORE_DRIVER_EXIT
  exit 143
fi
exit 99'''
    result = subprocess.run(
        [str(git_bash), "-lc", fixture],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 143, result.stdout + result.stderr
    assert "TERM_BEFORE_DRIVER_START" in result.stdout
    assert "TERM_BEFORE_DRIVER_EXIT" in result.stdout
    assert "WAITING_FOR_DRIVER_CLEAN_STOP" not in result.stdout + result.stderr


def test_bash_term_during_tee_drain_reaps_tee_once() -> None:
    if _skip_bash_fixture_on_windows("term_during_tee_drain"):
        return
    git_bash = _resolve_bash()
    fixture = r'''set -euo pipefail
fixture_dir="$(mktemp -d)"
trap 'rm -rf -- "$fixture_dir"' EXIT
fifo="$fixture_dir/tee.fifo"
log="$fixture_dir/tee.log"
mkfifo -- "$fifo"
tee "$log" < "$fifo" &
tee_pid=$!
driver_pid=''
term_requested=0
forward_term() {
  term_requested=1
  if [[ -n "$driver_pid" ]]; then
    kill -TERM -- "-$driver_pid" 2>/dev/null || kill -TERM "$driver_pid" 2>/dev/null || true
  elif [[ -n "$tee_pid" ]]; then
    echo TERM_DURING_TEE_DRAIN
  fi
}
trap forward_term TERM INT
bash -c 'printf tee-payload; sleep 1' > "$fifo" &
writer_pid=$!
(sleep 0.05; kill -TERM "$$") &
set +e
wait "$tee_pid"
first_tee_status=$?
if (( term_requested && (first_tee_status == 143 || first_tee_status == 130) )); then
  wait "$tee_pid"
  tee_status=$?
else
  tee_status="$first_tee_status"
fi
wait "$writer_pid"
set -e
grep -Fq tee-payload "$log"
printf '%s\n' "first_tee_status=$first_tee_status tee_status=$tee_status"
[[ "$first_tee_status" == 143 ]]
[[ "$tee_status" == 0 ]]'''
    result = subprocess.run(
        [str(git_bash), "-lc", fixture],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "first_tee_status=143 tee_status=0" in result.stdout


def test_bash_term_after_tee_before_driver_cleans_without_launch() -> None:
    if _skip_bash_fixture_on_windows("term_after_tee_before_driver"):
        return
    git_bash = _resolve_bash()
    # This fixture isolates the launcher race from FIFO-open semantics: the
    # real FIFO/tee drain is covered above and by static launcher assertions.
    # Here a directly killable sleep stands in for tee after its artifacts exist.
    fixture = r'''set -euo pipefail
fixture_dir="$(mktemp -d)"
fifo="$fixture_dir/pre_driver.fifo"
log="$fixture_dir/pre_driver.log"
mkfifo -- "$fifo"
: > "$log"
tee_pid=''
cleanup() {
  trap - EXIT
  if [[ -n "$tee_pid" ]]; then
    kill "$tee_pid" 2>/dev/null || true
    wait "$tee_pid" 2>/dev/null || true
  fi
  rm -f -- "$fifo" "$log"
  rmdir -- "$fixture_dir" 2>/dev/null || rm -rf -- "$fixture_dir"
}
trap cleanup EXIT
sleep 30 </dev/null >/dev/null 2>&1 &
tee_pid=$!
driver_pid=''
term_requested=0
forward_term() {
  term_requested=1
  if [[ -n "$driver_pid" ]]; then
    echo DRIVER_LAUNCHED
    kill -TERM "$driver_pid" 2>/dev/null || true
  elif [[ -n "$tee_pid" ]]; then
    echo TERM_AFTER_TEE_BEFORE_DRIVER
  fi
}
    trap forward_term TERM INT
forward_term
if (( term_requested )) && [[ -z "$driver_pid" ]]; then
  trap - TERM INT
  kill "$tee_pid" 2>/dev/null || true
  wait "$tee_pid" 2>/dev/null || true
  echo PRE_DRIVER_EXIT_143
  exit 143
fi
exit 99'''
    assert 'if (( term_requested )) && [[ -z "$driver_pid" ]]; then' in fixture
    result = subprocess.run(
        [str(git_bash), "-lc", fixture],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 143, result.stdout + result.stderr
    assert "TERM_AFTER_TEE_BEFORE_DRIVER" in result.stdout
    assert "PRE_DRIVER_EXIT_143" in result.stdout
    assert "DRIVER_LAUNCHED" not in result.stdout


def main() -> int:
    test_returned_143_is_not_a_failure()
    test_raised_system_exit_is_distinct()
    test_clean_report_and_checkpoint_evidence_are_validated()
    test_compute_cap_report_is_nonterminal_signal_only_and_stage2_free()
    test_term_at_epoch30_prefers_terminal_cap_boundary()
    test_stage1_clean_latest_resume_requires_companion_record()
    test_compute_cap_resume_is_total_epoch_30_and_requires_clean_source()
    test_compute_cap_boundary_rejects_final_bundle_and_stage2()
    test_terminal_final_recovery_is_zero_update()
    test_original_metrics_and_run_wide_resources_are_persisted()
    test_signal_forwarding_reaches_cooperative_handlers()
    test_launcher_rejects_lone_stage1_latest()
    test_resource_postflight_validation_is_executable()
    test_resume_helper_accepts_bundle_before_final_fallback()
    test_repeated_stage2_resume_stop_preserves_lifecycle_latest()
    test_stage2_stop_seam_preserves_existing_flag_and_default_resets()
    test_bash_wait_fixture_reaps_after_interrupted_wait()
    test_bash_term_before_driver_exits_without_wait_spin()
    test_bash_term_during_tee_drain_reaps_tee_once()
    test_bash_term_after_tee_before_driver_cleans_without_launch()
    if os.name == "nt":
        print(
            "SE_B7873200_FULL_BASH_FIXTURE_PACE_LINUX_REQUIRED: "
            f"skipped={BASH_FIXTURE_SKIP_COUNT}"
        )
    print("SE_B7873200_FULL_CONTROL_TEST_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
