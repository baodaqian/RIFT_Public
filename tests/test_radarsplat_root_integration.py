"""Shared root integration checks, separate from independent owner suites."""
from pathlib import Path


def test_rift_shared_root_calls_current_owned_implementation(tmp_path):
    import train_rift_dataset as shared
    import train_radarsplat as trainer
    from rift.radarsplat_collection import commands_for
    kw=dict(dataset_root=tmp_path/'data',output_root=tmp_path/'runs')
    generated=shared.commands_for('a320','radarsplat',**kw)
    assert generated==commands_for('a320',**kw)
    args=trainer.parse_args(generated[-1][2:])
    assert args.fidelity_profile=='budget48' and args.steps==2000 and args.init_num_gaussians==112000


def test_gotcha_shared_root_calls_current_owned_backend(tmp_path,monkeypatch):
    import train_gotcha_dataset as shared
    from rift import radarsplat_gotcha as backend
    spec=shared.backend_registry()['radarsplat']
    assert spec['status']=='available' and spec['native_frequency_policy']=='ragged_exact'
    seen={}
    def fake(**kw): seen.update(kw); return {'status':'complete'}
    monkeypatch.setattr(backend,'run_gotcha',fake)
    dataset=object()
    result=shared.dispatch(dataset,dict(method='radarsplat',backend=spec,config={},output_dir=str(tmp_path),resume=None),'cpu')
    assert result['status']=='complete' and seen['dataset'] is dataset
    assert seen['output_dir']==Path(tmp_path) and seen['config']=={}
