"""Fail-closed A100 runtime validation for the Camry full-polarization core."""

from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import unittest
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CORE_TEST_COUNT = 19
REQUIRED_GPU_LABEL = "A100"
REQUIRED_GPU_TOKEN = "a100"
REQUIRED_CUDA_TEST = (
    "tests.test_gotcha_camry_fullpol_core.CamryTrainingContractTest."
    "test_cuda_checkpoint_round_trip_restores_rng_optimizer_devices_and_resume_readiness"
)


class _RecordingResult(unittest.TextTestResult):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.successful_ids: list[str] = []

    def addSuccess(self, test: unittest.case.TestCase) -> None:  # noqa: N802 - unittest API
        super().addSuccess(test)
        self.successful_ids.append(test.id())


def _iter_cases(suite: unittest.TestSuite) -> Iterable[unittest.case.TestCase]:
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _iter_cases(item)
        else:
            yield item


def _failure_ids(items: Iterable[tuple[unittest.case.TestCase, str]]) -> list[str]:
    return [test.id() for test, _traceback in items]


def _test_details(items: Iterable[tuple[unittest.case.TestCase, str]]) -> list[dict[str, str]]:
    return [{"id": test.id(), "traceback": traceback} for test, traceback in items]


def _report(**values: Any) -> dict[str, Any]:
    result = {"schema": "rift_gotcha_camry_fullpol_runtime_validation_v1", **values}
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> int:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    try:
        import torch
    except Exception as exc:
        _report(
            status="FAIL_RUNTIME",
            reason="torch_unavailable" if isinstance(exc, ModuleNotFoundError) else "torch_import_failed",
            detail=str(exc),
            expected_core_test_count=EXPECTED_CORE_TEST_COUNT,
            discovered_core_test_count=None,
            skipped=0,
            failures=0,
            errors=0,
            cuda_restore_resume_test_passed=False,
        )
        return 2

    try:
        cuda_available = bool(torch.cuda.is_available())
        device_count = int(torch.cuda.device_count())
        device_names = tuple(str(torch.cuda.get_device_name(index)) for index in range(device_count))
        selected_device_index = int(torch.cuda.current_device())
        selected_device_name = str(torch.cuda.get_device_name(selected_device_index))
    except Exception as exc:
        _report(
            status="FAIL_RUNTIME",
            reason="cuda_probe_failed",
            detail=f"{type(exc).__name__}: {exc}",
            expected_core_test_count=EXPECTED_CORE_TEST_COUNT,
            discovered_core_test_count=None,
            skipped=0,
            failures=0,
            errors=1,
            cuda_restore_resume_test_passed=False,
        )
        return 2
    if not cuda_available:
        _report(
            status="FAIL_RUNTIME",
            reason="cuda_unavailable",
            expected_core_test_count=EXPECTED_CORE_TEST_COUNT,
            discovered_core_test_count=None,
            skipped=0,
            failures=0,
            errors=0,
            cuda_restore_resume_test_passed=False,
        )
        return 2
    if device_count != 1:
        _report(
            status="FAIL_RUNTIME",
            reason="exactly_one_cuda_device_required",
            device_names=device_names,
            selected_device_index=selected_device_index,
            selected_device_name=selected_device_name,
            expected_gpu=REQUIRED_GPU_LABEL,
            expected_core_test_count=EXPECTED_CORE_TEST_COUNT,
            discovered_core_test_count=None,
            skipped=0,
            failures=0,
            errors=0,
            cuda_restore_resume_test_passed=False,
        )
        return 2
    if not selected_device_name or REQUIRED_GPU_TOKEN not in selected_device_name.lower():
        _report(
            status="FAIL_RUNTIME",
            reason="a100_required",
            device_names=device_names,
            selected_device_index=selected_device_index,
            selected_device_name=selected_device_name,
            expected_gpu=REQUIRED_GPU_LABEL,
            expected_core_test_count=EXPECTED_CORE_TEST_COUNT,
            discovered_core_test_count=None,
            skipped=0,
            failures=0,
            errors=0,
            cuda_restore_resume_test_passed=False,
        )
        return 2

    try:
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromName("tests.test_gotcha_camry_fullpol_core")
        cases = tuple(_iter_cases(suite))
        stream = io.StringIO()
        runner = unittest.TextTestRunner(stream=stream, verbosity=0, resultclass=_RecordingResult)
        result = runner.run(suite)
        discovered_ids = tuple(test.id() for test in cases)
        failure_ids = _failure_ids(result.failures)
        error_ids = _failure_ids(result.errors)
        skipped_ids = [test.id() for test, _reason in result.skipped]
        cuda_restore_resume_test_passed = REQUIRED_CUDA_TEST in result.successful_ids
        test_diagnostics: dict[str, Any] = {}
        if result.failures:
            test_diagnostics["failure_details"] = _test_details(result.failures)
        if result.errors:
            test_diagnostics["error_details"] = _test_details(result.errors)
        if result.failures or result.errors:
            test_diagnostics["test_runner_output"] = stream.getvalue()
        report = _report(
            status="PASS_RUNTIME" if (
                len(cases) == EXPECTED_CORE_TEST_COUNT
                and result.testsRun == EXPECTED_CORE_TEST_COUNT
                and not result.skipped
                and not result.failures
                and not result.errors
                and cuda_restore_resume_test_passed
            ) else "FAIL_RUNTIME",
            device_names=device_names,
            selected_device_index=selected_device_index,
            selected_device_name=selected_device_name,
            expected_gpu=REQUIRED_GPU_LABEL,
            expected_core_test_count=EXPECTED_CORE_TEST_COUNT,
            discovered_core_test_count=len(cases),
            tests_run=int(result.testsRun),
            skipped=len(result.skipped),
            failures=len(result.failures),
            errors=len(result.errors),
            skipped_ids=skipped_ids,
            failure_ids=failure_ids,
            error_ids=error_ids,
            cuda_restore_resume_test=REQUIRED_CUDA_TEST,
            cuda_restore_resume_test_discovered=REQUIRED_CUDA_TEST in discovered_ids,
            cuda_restore_resume_test_passed=cuda_restore_resume_test_passed,
            **test_diagnostics,
        )
        return 0 if report["status"] == "PASS_RUNTIME" else 2
    except Exception as exc:  # The validator must fail closed with a machine-readable result.
        _report(
            status="FAIL_RUNTIME",
            reason="validator_exception",
            detail=f"{type(exc).__name__}: {exc}",
            device_names=device_names,
            selected_device_index=selected_device_index,
            selected_device_name=selected_device_name,
            expected_gpu=REQUIRED_GPU_LABEL,
            expected_core_test_count=EXPECTED_CORE_TEST_COUNT,
            discovered_core_test_count=None,
            skipped=0,
            failures=0,
            errors=1,
            cuda_restore_resume_test_passed=False,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
