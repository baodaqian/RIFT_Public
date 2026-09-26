"""Merged GOTCHA routing: synthetic HH metadata and mocked execution only."""
import json
from pathlib import Path
import sys

import pytest

import train_gotcha_dataset as cli
from rift.gotcha_dataset import NativeShardReader
from tests.se_dataset_fixtures import tiny_region, write_shard


@pytest.fixture
def argv(tmp_path, monkeypatch):
    root = tmp_path/'data'
    write_shard(root/'New_Transfer'/'shards'/'pass1_hh.npz', nf=11)
    monkeypatch.setattr(cli, 'load_region', lambda *_: tiny_region())
    monkeypatch.setattr(NativeShardReader, 'read', lambda *_: pytest.fail('Response access during routing'))
    return ['--dataset-root', str(root), '--passes', '1', '--output-root', str(tmp_path/'runs')]


def test_joint_hh_plan_preserves_both_owner_recipes(argv, tmp_path):
    config = tmp_path/'methods.json'
    config.write_text(json.dumps({'radarsplat': {'point_chunk': 17},
                                 'sugavanam_ertin': {'stage1_iterations': 19}}))
    args = cli.parse_args(argv+['--method', 'rs', 'se', '--method-config', str(config)])
    dataset, report = cli.make_plan(args)
    rs, se = report['plans']
    assert args.polarizations == ('hh',)
    assert rs['method'] == 'radarsplat' and se['method'] == 'sugavanam_ertin'
    assert rs['config']['point_chunk'] == 17 and 'stage1_iterations' not in rs['config']
    assert rs['recipe']['model_updates_per_head'] == 2000
    assert rs['targets_per_head']['hh']['train'] == 250
    assert se['config'] == {'stage1_iterations': 19}
    assert se['se']['recipe']['azimuth_bins'] == 72
    assert se['se']['recipe']['stage1_iterations'] == 19
    assert 'unresolved' in se['fidelity_status']
    assert Path(rs['output_dir']).parent == Path(se['output_dir']).parent
    assert rs['output_dir'] != se['output_dir']
    assert Path(rs['resume_file']).name == 'radarsplat_gotcha.pt'
    assert report['dataset']['viewpoints']['test'] == 55
    assert not report['dataset']['response_payload_read']
    assert sum(s.response_reads for s in dataset.shards.values()) == 0
    assert not args.output_root.exists()


@pytest.mark.parametrize('method,alias', [('radarsplat', 'rs'), ('sugavanam_ertin', 'se')])
def test_single_backend_config_and_method_mapping_produce_same_plan(argv, tmp_path, method, alias):
    config = tmp_path/'config.json'
    config.write_text('{}')
    mapping = tmp_path/'mapping.json'
    mapping.write_text(json.dumps({method: {}}))
    old_args = cli.parse_args(argv+['--method', alias, '--method-config', str(mapping)])
    dataset, old = cli.make_plan(old_args)
    shared_dataset, shared = cli.make_plan(cli.parse_args(argv+['--method', method, '--config', str(config)]))
    assert old == shared and dataset.contract == shared_dataset.contract
    assert cli.parse_args([]).output_root.name == 'GOTCHA_dataset'
    assert old_args.polarizations == ('hh',)


@pytest.mark.parametrize('extra,match', [
    (['--method', 'rs', 'se', '--config', 'unused.json'], 'exactly one baseline'),
    (['--method', 'rs', 'se', '--resume', 'unused.pt'], 'one learned method'),
    (['--method', 'rs', 'se', '--check-initialization'], 'alone'),
    (['--method', 'se', '--polarizations', 'vv'], 'polarization'),
])
def test_ambiguous_options_fail_before_dataset_access(monkeypatch, extra, match):
    monkeypatch.setattr(cli, 'GOTCHADataset', lambda *a, **kw: pytest.fail('Invalid options opened data'))
    with pytest.raises(ValueError, match=match):
        cli.make_plan(cli.parse_args(extra))


@pytest.mark.parametrize('config', [
    {'radarsplat': {'steps': 3}},
    {'sugavanam_ertin': {'azimuth_bins': 2}},
    {'radarsplat': []},
])
def test_owner_config_validation_precedes_data_access(tmp_path, monkeypatch, config):
    path = tmp_path/'config.json'
    path.write_text(json.dumps(config))
    monkeypatch.setattr(cli, 'GOTCHADataset', lambda *a, **kw: pytest.fail('Invalid recipe opened data'))
    with pytest.raises(ValueError):
        cli.make_plan(cli.parse_args(['--method', 'rs', 'se', '--method-config', str(path)]))


def test_both_actual_root_hooks_receive_same_dataset_and_separate_configs(argv, tmp_path, monkeypatch):
    from rift import radarsplat_gotcha as rs, sugavanam_ertin_paper_workflow as se
    calls = []
    def capture(method):
        def run(**kw):
            calls.append((method, kw))
            return {'status': 'complete'}
        return run
    monkeypatch.setattr(rs, 'run_gotcha', capture('radarsplat'))
    monkeypatch.setattr(se, 'run_gotcha', capture('sugavanam_ertin'))
    monkeypatch.setenv('SLURM_JOB_ID', 'synthetic_routing_fixture')
    cli.main(argv+['--method', 'rs', 'se', '--device', 'cpu'])
    assert [m for m, _ in calls] == ['radarsplat', 'sugavanam_ertin']
    first, second = [kw for _, kw in calls]
    assert first['dataset'] is second['dataset']
    assert first['dataset'].polarizations == ('hh',)
    assert first['output_dir'].name == 'radarsplat' and second['output_dir'].name == 'sugavanam_ertin'
    assert first['device'] == second['device'] == 'cpu'
    assert first['resume'] is second['resume'] is None
    assert second['config'] == {} and first['config']['azimuth_samples'] == 33
    assert not (tmp_path/'runs').exists()


def test_all_destinations_are_checked_before_first_method(argv, monkeypatch):
    args = argv+['--method', 'rs', 'se']
    _, report = cli.make_plan(cli.parse_args(args))
    occupied = Path(report['plans'][1]['output_dir'])
    occupied.mkdir(parents=True)
    (occupied/'checkpoint_latest.pt').write_text('existing fixture')
    monkeypatch.setenv('SLURM_JOB_ID', 'synthetic_routing_fixture')
    monkeypatch.setattr(cli, 'dispatch', lambda *a: pytest.fail('Started before preflighting all outputs'))
    with pytest.raises(ValueError, match='Existing output'):
        cli.main(args)


def test_probe_has_nonzero_exit_without_dispatch_or_output(argv, monkeypatch, tmp_path, capsys):
    from rift import gotcha_baseline_planning as planning
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    monkeypatch.setattr(planning, 'check_initialization', lambda *a: {'status': 'initialization_degenerate'})
    monkeypatch.setattr(cli, 'dispatch', lambda *a: pytest.fail('Initialization probe attempted fitting'))
    assert cli.main(argv+['--method', 'se', '--check-initialization']) == 2
    report = json.loads(capsys.readouterr().out)
    assert report['plans'][0]['initialization_audit']['status'] == 'initialization_degenerate'
    assert not (tmp_path/'runs').exists()


def test_incomplete_se_still_fails_and_prevents_following_method(argv, monkeypatch):
    from rift import sugavanam_ertin_paper_workflow as se, radarsplat_gotcha as rs
    monkeypatch.setenv('SLURM_JOB_ID', 'synthetic_routing_fixture')
    monkeypatch.setattr(se, 'run', lambda *a, **kw: {'status': 'initialization_degenerate'})
    monkeypatch.setattr(rs, 'run_gotcha', lambda **kw: pytest.fail('Continued after failed SE'))
    with pytest.raises(RuntimeError, match='SE comparison incomplete: initialization_degenerate'):
        cli.main(argv+['--method', 'se', 'rs'])


@pytest.mark.parametrize('method', ['radarsplat', 'sugavanam_ertin'])
def test_resume_and_shared_interruption_code(argv, monkeypatch, tmp_path, method):
    monkeypatch.setenv('SLURM_JOB_ID', 'synthetic_routing_fixture')
    checkpoint = tmp_path/'resume.pt'
    checkpoint.write_text('placeholder; routing only')
    calls = []
    def stop(dataset, entry, device):
        calls.append(entry)
        return {'status': 'interrupted'}
    monkeypatch.setattr(cli, 'dispatch', stop)
    with pytest.raises(SystemExit) as exc:
        cli.main(argv+['--method', method, '--resume', str(checkpoint)])
    assert exc.value.code == 143
    assert len(calls) == 1 and calls[0]['method'] == method and calls[0]['resume'] == str(checkpoint)


def test_list_imports_no_training_engine(monkeypatch, capsys):
    for module in ('rift.radarsplat_gotcha', 'rift.sugavanam_ertin_paper_workflow'):
        monkeypatch.setitem(sys.modules, module, None)
    monkeypatch.setattr(cli, 'GOTCHADataset', lambda *a, **kw: pytest.fail('List opened data'))
    cli.main(['--list'])
    registry = json.loads(capsys.readouterr().out)
    assert registry['radarsplat']['status'] == registry['sugavanam_ertin']['status'] == 'available'
    assert registry['sugavanam_ertin']['polarizations'] == ['hh']
