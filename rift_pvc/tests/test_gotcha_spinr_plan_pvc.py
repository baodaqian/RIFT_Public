"""The PVC GOTCHA frontend plans SpINR with the PVC preflight, so the plan discloses pulse_execution.

Audit finding (2026-09-22): ``train_gotcha_dataset_pvc.make_plan`` imported the CUDA
planner's ``preflight``, whose recipe has no ``pulse_execution``, while the runtime
(``rift_pvc.spinr_gotcha_training.run``) selects the batched GEMM lane. The plan must
carry the same recipe the runtime writes to ``recipe.json``.
"""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import train_gotcha_dataset_pvc as frontend  # noqa: E402
from rift import spinr_gotcha_training as original_runtime  # noqa: E402
from rift_pvc import spinr_gotcha_training as runtime  # noqa: E402
from rift_pvc.spinr_native_batched import PULSE_EXECUTION  # noqa: E402
from tests.test_spinr_gotcha import config, native_dataset  # noqa: E402,F401 (fixture)


@pytest.fixture(autouse=True)
def cpu_backend(monkeypatch):
    monkeypatch.setenv('RIFT_ACCELERATOR', 'cpu')


def test_spinr_plan_discloses_the_pvc_pulse_execution(monkeypatch, tmp_path, native_dataset):
    monkeypatch.setattr(frontend, 'GOTCHADataset', lambda *args, **kwargs: native_dataset)
    monkeypatch.setattr(frontend, 'load_region', lambda name, path: native_dataset.region)
    method_config = tmp_path/'spinr.json'
    method_config.write_text(json.dumps({'spinr': config()}))
    args = frontend.parse_args(['--method', 'spinr', '--method-config', str(method_config),
                                '--output-root', str(tmp_path/'out'), '--dry-run'])
    dataset, plan = frontend.make_plan(args)
    assert dataset is native_dataset
    entry, = plan['plans']
    assert entry['method'] == 'spinr' and entry['backend']['module'] == 'train_spinr_style_pvc'
    native_plan = entry['native_plan']
    recipe = native_plan['recipe']
    # The plan carries the runtime's disclosure, and nothing else differs from the CUDA planner.
    assert recipe['pulse_execution'] == PULSE_EXECUTION
    assert recipe == runtime.preflight(native_dataset, entry['config'])['recipe']
    expected = original_runtime.preflight(native_dataset, entry['config'])
    assert {k: v for k, v in recipe.items() if k != 'pulse_execution'} == expected['recipe']
    assert {k: v for k, v in native_plan.items() if k != 'recipe'} == {k: v for k, v in expected.items() if k != 'recipe'}
    assert native_plan['response_reads'] is False and native_plan['test_accessed'] is False
    assert json.loads(json.dumps(plan))['plans'][0]['native_plan']['recipe']['pulse_execution'] == PULSE_EXECUTION
