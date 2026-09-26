"""Merged collection frontend routing, without real conversion or fitting."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import train_rift_dataset as cli


@pytest.fixture
def metadata(monkeypatch):
    calls = []
    def preflight(root, name, num_train, antenna_selection=None):
        calls.append(name)
        return {'object_id': name}
    monkeypatch.setattr(cli, 'preflight_object', preflight)
    return calls


@pytest.mark.parametrize('method,alias', [('radarsplat', 'rs'), ('sugavanam_ertin', 'se')])
def test_canonical_aliases_preserve_same_commands(metadata, tmp_path, method, alias):
    argv = ['--object', 'loader', '--output-root', str(tmp_path), '--device', 'cpu']
    old = cli.make_plan(cli.parse_args(argv+['--method', alias]))
    shared = cli.make_plan(cli.parse_args(argv+['--method', method]))
    assert old == shared
    assert old['plans'][0]['dataset_identity'] == {'object_id': 'loader'}
    assert cli.parse_args([]).output_root.name == 'RIFT_dataset'


def test_both_methods_all_objects_config_goes_only_to_se(metadata, tmp_path):
    config = tmp_path/'se.json'
    config.write_text(json.dumps({'stage1_iterations': 19}))
    args = cli.parse_args(['--method', 'rs', 'se', '--se-config', str(config),
                           '--output-root', str(tmp_path/'runs'), '--device', 'cuda'])
    report = cli.make_plan(args)
    assert len(metadata) == 6 and len(report['plans']) == 12
    for entry in report['plans']:
        assert entry['dataset_identity']['object_id'] == entry['object']
        assert Path(entry['output_dir']).parent.name == entry['object']
        for command in entry['commands']:
            assert command[command.index('--device')+1] == 'cuda'
            assert ('--config' in command) == (entry['method'] == 'sugavanam_ertin')
        if entry['method'] == 'sugavanam_ertin':
            assert entry['commands'][0][entry['commands'][0].index('--config')+1] == str(config)
        else:
            assert entry['recipe'] == 'budget48'
    assert not (tmp_path/'runs').exists()


def test_config_and_probe_fail_before_metadata_for_incompatible_recipe(metadata, tmp_path):
    config = tmp_path/'bad.json'
    config.write_text(json.dumps({'hidden_dim': 16}))
    for extra in (['--method', 'se', '--config', str(config)],
                  ['--method', 'se', '--se-recipe', 'legacy-full', '--check-initialization'],
                  ['--method', 'rs', 'se', '--check-initialization']):
        with pytest.raises(ValueError):
            cli.make_plan(cli.parse_args(extra))
    assert metadata == []


def test_dependency_failure_happens_before_any_target_preparation(metadata, tmp_path, monkeypatch):
    from rift import radarsplat_release
    monkeypatch.setenv('SLURM_JOB_ID', 'synthetic_routing_fixture')
    monkeypatch.setattr(cli.subprocess, 'run', lambda *a, **kw: pytest.fail('Started before CUDA preflight'))
    def missing(**kw):
        raise RuntimeError('synthetic missing source CUDA')
    monkeypatch.setattr(radarsplat_release, 'load_cuda_reference', missing)
    with pytest.raises(RuntimeError, match='synthetic missing source CUDA'):
        cli.main(['--object', 'loader', '--method', 'se', 'rs', '--output-root', str(tmp_path)])


def test_later_destination_preflight_blocks_all_launches(metadata, tmp_path, monkeypatch):
    from rift.antenna_selection import acquisition_label, selection
    occupied = tmp_path/'train2400'/acquisition_label(selection(1,1))/'loader'/'sugavanam_ertin'
    occupied.mkdir(parents=True)
    (occupied/'checkpoint.pt').write_text('fixture')
    monkeypatch.setenv('SLURM_JOB_ID', 'synthetic_routing_fixture')
    monkeypatch.setattr(cli.subprocess, 'run', lambda *a, **kw: pytest.fail('Started before all destinations checked'))
    with pytest.raises(ValueError, match='Existing output'):
        cli.main(['--object', 'a320', 'loader', '--method', 'rs', 'se', '--output-root', str(tmp_path)])


def test_shared_sequence_preserves_commands_and_stops_on_nonzero_exit(metadata, tmp_path, monkeypatch):
    from rift import radarsplat_release
    monkeypatch.setenv('SLURM_JOB_ID', 'synthetic_routing_fixture')
    preflight = []
    monkeypatch.setattr(radarsplat_release, 'load_cuda_reference', lambda **kw: preflight.append(kw))
    calls = []
    def child(command, **kwargs):
        assert len(preflight) == 1
        calls.append(Path(command[1]).name)
        return SimpleNamespace(returncode=2 if len(calls) == 3 else 0)
    monkeypatch.setattr(cli.subprocess, 'run', child)
    status = cli.main(['--object', 'a320', 'loader', '--method', 'rs', 'se', '--output-root', str(tmp_path)])
    assert status == 2
    assert calls == ['prepare_radarsplat_b7873200_targets.py', 'train_radarsplat.py', 'train_sugavanam_ertin.py']


def test_dry_run_skips_cuda_and_subprocess(metadata, tmp_path, monkeypatch):
    from rift import radarsplat_release
    monkeypatch.setattr(radarsplat_release, 'load_cuda_reference', lambda **kw: pytest.fail('dry-run touched CUDA'))
    monkeypatch.setattr(cli.subprocess, 'run', lambda *a, **kw: pytest.fail('dry-run launched work'))
    assert cli.main(['--object', 'a320', '--method', 'rs', 'se', '--dry-run',
                     '--output-root', str(tmp_path/'runs')]) == 0
    assert not (tmp_path/'runs').exists()


def test_resume_contracts_remain_method_specific(metadata, tmp_path):
    common = ['--object', 'a320', '--output-root', str(tmp_path)]
    report = cli.make_plan(cli.parse_args(common+['--method', 'rs', '--resume', 'auto']))
    assert '--resume' in report['plans'][0]['commands'][-1]
    with pytest.raises(ValueError, match='explicit'):
        cli.make_plan(cli.parse_args(common+['--method', 'se', '--resume', 'auto']))
    with pytest.raises(ValueError, match='one object and one method'):
        cli.make_plan(cli.parse_args(common+['--method', 'rs', 'se', '--resume', 'auto']))
