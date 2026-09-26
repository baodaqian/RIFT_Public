"""Production scheduling boundaries preserve recipes and fail scientific errors."""
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import train_gotcha_dataset_pvc as frontend
from rift_pvc import sugavanam_ertin_paper_workflow as workflow


@pytest.mark.parametrize('status', ['stage1_complete', 'interrupted',
                                   'stage1_unconverged', 'initialization_degenerate'])
def test_gotcha_stage_boundary_keeps_recipe_and_failure_status(monkeypatch, tmp_path, status):
    import rift.sugavanam_ertin_stage2_runtime_v1 as runtime
    monkeypatch.setenv('SLURM_JOB_ID', 'test')
    monkeypatch.setattr(runtime, 'install_stop_handlers', lambda: None)
    acquisition = SimpleNamespace(kind='gotcha_native')
    monkeypatch.setattr(workflow, 'GOTCHAAcquisition', lambda dataset: acquisition)
    seen = {}
    def run(acq, recipe, output, **kwargs):
        seen.update(acq=acq, recipe=recipe, output=output, **kwargs)
        return {'status': status}
    monkeypatch.setattr(workflow, 'run', run)
    config = {'granularity': 40, 'export_grid': 48, 'initialization_std': .05}
    kwargs = dict(dataset=object(), output_dir=tmp_path, config=config,
                  device='xpu', resume=tmp_path/'checkpoint_latest.pt')
    if status in ('stage1_complete', 'interrupted'):
        assert workflow.run_gotcha_stage1(**kwargs)['status'] == status
    else:
        with pytest.raises(RuntimeError, match=status):
            workflow.run_gotcha_stage1(**kwargs)
    assert seen['recipe'] == workflow.make_recipe('gotcha_native', config)
    assert seen['recipe']['stage2_steps'] == 5000
    assert seen['stage1_only'] is True
    assert seen['resume'] == kwargs['resume']


def test_frontend_routes_stage_boundary_with_original_dataset(monkeypatch, tmp_path):
    dataset = object()
    seen = {}
    monkeypatch.setattr(workflow, 'run_gotcha_stage1', lambda **kw: seen.update(kw) or {'status': 'stage1_complete'})
    entry = dict(method='sugavanam_ertin', backend=frontend.backend_registry()['sugavanam_ertin'],
                 config={}, output_dir=str(tmp_path), resume=None, stage1_only=True)
    assert frontend.dispatch(dataset, entry, 'xpu')['status'] == 'stage1_complete'
    assert seen['dataset'] is dataset
    assert seen['config'] == {} and seen['device'] == 'xpu'


def test_stage_boundary_rejects_other_methods_before_data_access():
    args = frontend.parse_args(['--method', 'rift', '--se-stage1-only'])
    with pytest.raises(ValueError, match='requires --method sugavanam_ertin alone'):
        frontend.make_plan(args)
