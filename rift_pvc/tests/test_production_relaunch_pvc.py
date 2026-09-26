"""Wall-clock resume in scripts_pvc/production_relaunch.py: collection GeRaF and the GOTCHA frontend (fake Slurm, temp campaign)."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts_pvc'))
import production_relaunch as relaunch  # noqa: E402

JOB = 2155387
ROW = dict(state='TIMEOUT', start='2026-09-22T10:45:02', end='2026-09-23T10:45:05', elapsed='1-00:00:03', nodes='ac078')


def make_root(tmp_path, *, checkpoint=True, status_rc=143, fallback=False, with_status=True,
              script='train_rift_dataset_pvc.py', method='geraf', resume_in_command=False):
    root = tmp_path/'campaign'
    for name in ('status', 'logs', 'source'):
        (root/name).mkdir(parents=True)
    out = root/'outputs/rift/train2400/1t1r/race_car/geraf/source_v1'
    out.mkdir(parents=True)
    if checkpoint:
        (out/'checkpoints').mkdir()
        (out/'checkpoints/checkpoint_latest.pth.tar').write_bytes(b'ckpt')
    command = ['/env/bin/python', '-u', str(root/'source'/script), '--object', 'racecar', '--method', method]
    if resume_in_command:
        command += ['--resume', 'auto']
    task = dict(key='racecar-geraf-full', scene='racecar', method=method, stage='full', output_dir=str(out),
                command=command, depends_on=None, job_name='pvcprod-racecar-geraf-full', job_id=JOB)
    if with_status:
        (root/'status'/f'racecar-geraf-full-{JOB}.json').write_text(json.dumps(
            dict(task='racecar-geraf-full', job_id=str(JOB), returncode=status_rc, fallback=fallback)))
    (root/'campaign.json').write_text(json.dumps(dict(tasks=[task])))
    return root, task


def test_timeout_with_checkpoint_resumes_auto(tmp_path):
    root, task = make_root(tmp_path, with_status=False)
    kind, reason = relaunch.classify(task, {JOB: ROW}, {}, root)
    assert kind == 'resume-auto' and 'checkpoint_latest.pth.tar' in reason and 'TIMEOUT' in reason


def test_failed_with_clean_sigterm_status_resumes_auto(tmp_path):
    root, task = make_root(tmp_path, status_rc=143)
    kind, _ = relaunch.classify(task, {JOB: dict(ROW, state='FAILED')}, {}, root)
    assert kind == 'resume-auto'


@pytest.mark.parametrize('kwargs, message', [
    (dict(status_rc=1), 'not the clean SIGTERM'),
    (dict(fallback=True), 'XPU fallback'),
    (dict(checkpoint=False, with_status=False), 'missing'),
    (dict(resume_in_command=True), 'already carries --resume'),
    # GeRaF on the GOTCHA frontend resumes <output>/checkpoint_latest.pth.tar, not the collection lane's
    # checkpoints/ file this fixture writes (see the GOTCHA tests below).
    (dict(script='train_gotcha_dataset_pvc.py'), 'checkpoint_latest.pth.tar is missing'),
])
def test_refusals(tmp_path, kwargs, message):
    root, task = make_root(tmp_path, **kwargs)
    with pytest.raises(RuntimeError, match=message):
        relaunch.classify(task, {JOB: ROW}, {}, root)


def test_failed_without_status_report_is_refused(tmp_path):
    root, task = make_root(tmp_path, with_status=False)
    with pytest.raises(RuntimeError, match='without a runner status report'):
        relaunch.classify(task, {JOB: dict(ROW, state='FAILED')}, {}, root)


def test_non_geraf_timeout_still_refused(tmp_path):
    root, task = make_root(tmp_path, method='rift', with_status=False)
    with pytest.raises(RuntimeError, match='not cancelled'):
        relaunch.classify(task, {JOB: ROW}, {}, root)


def test_classify_keeps_three_argument_call(tmp_path):
    root, task = make_root(tmp_path, with_status=False)
    assert relaunch.classify(task, {JOB: ROW}, {})[0] == 'resume-auto'


def test_relaunch_submits_resume_auto_and_archives(tmp_path, monkeypatch, capsys):
    root, task = make_root(tmp_path, with_status=False)
    monkeypatch.setattr(relaunch, 'accounting', lambda ids: {JOB: ROW})
    monkeypatch.setattr(relaunch, 'live_queue', lambda: {})
    relaunch.relaunch(root, ('racecar',), [], dry_run=True, keys=['racecar-geraf-full'])
    assert '--resume auto' in capsys.readouterr().out
    assert json.loads((root/'campaign.json').read_text())['tasks'][0]['job_id'] == JOB
    calls = []
    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout='2160000\n', stderr='')
    monkeypatch.setattr(relaunch.subprocess, 'run', fake_run)
    ledger = relaunch.relaunch(root, ('racecar',), [], dry_run=False, keys=['racecar-geraf-full'])
    saved = json.loads((root/'campaign.json').read_text())['tasks'][0]
    assert saved['job_id'] == 2160000 and saved['command'][-2:] == ['--resume', 'auto']
    assert saved['relaunch']['kind'] == 'resume-auto' and saved['relaunch']['previous_job_id'] == JOB
    assert saved['submission_history'][0]['job_id'] == JOB and saved['submission_history'][0]['command'] == task['command']
    assert calls[0][0] == 'sbatch' and str(root/'source/scripts_pvc/production_job.sbatch') in calls[0]
    assert ledger[0]['kind'] == 'resume-auto' and json.loads((root/'relaunches.json').read_text())[0]['jobs'][0]['job_id'] == 2160000


# --- GOTCHA frontend (train_gotcha_dataset_pvc.py) -----------------------------------------------------------

GOTCHA_DIR = 'outputs/gotcha/camry/f5d77de24a59fa0b/pulse_subset16_all_roles_v2/frequency_stride2_all_roles_v2'


def make_gotcha_root(tmp_path, method='rift', *, directory=None, resume_file=True, status_rc=143, fallback=False,
                     with_status=True, resume_arg=None):
    root = tmp_path/'campaign'
    for name in ('status', 'logs', 'source'):
        (root/name).mkdir(parents=True)
    out = root/GOTCHA_DIR/(directory or method)
    out.mkdir(parents=True)
    if resume_file and method in relaunch.GOTCHA_RESUME_FILES:
        (out/relaunch.GOTCHA_RESUME_FILES[method]).write_bytes(b'ckpt')
    key = f'camry-{method}-full'
    command = ['/env/bin/python', '-u', str(root/'source/train_gotcha_dataset_pvc.py'), '--region', 'camry',
               '--method', method, '--device', 'xpu']
    if resume_arg is not None:
        command += ['--resume', str(resume_arg(out))]
    task = dict(key=key, scene='camry', method=method, stage='full', output_dir=str(out), command=command,
                depends_on=None, job_name=f'pvcprod-{key}', job_id=JOB)
    if with_status:
        (root/'status'/f'{key}-{JOB}.json').write_text(json.dumps(
            dict(task=key, job_id=str(JOB), returncode=status_rc, fallback=fallback)))
    (root/'campaign.json').write_text(json.dumps(dict(tasks=[task])))
    return root, task, out


@pytest.mark.parametrize('method, name', sorted(relaunch.GOTCHA_RESUME_FILES.items()))
def test_gotcha_timeout_resumes_each_backends_own_file(tmp_path, method, name):
    root, task, out = make_gotcha_root(tmp_path, method, with_status=False)
    kind, reason = relaunch.classify(task, {JOB: ROW}, {}, root)
    assert kind == 'resume-gotcha' and name in reason and 'TIMEOUT' in reason and method in reason


def test_gotcha_failed_with_clean_sigterm_status_resumes(tmp_path):
    root, task, _ = make_gotcha_root(tmp_path, 'radarsplat', status_rc=143)
    assert relaunch.classify(task, {JOB: dict(ROW, state='FAILED')}, {}, root)[0] == 'resume-gotcha'


@pytest.mark.parametrize('method, kwargs, state, message', [
    ('rift', dict(status_rc=1), 'TIMEOUT', 'not the clean SIGTERM'),
    ('radar_fields', dict(fallback=True), 'FAILED', 'XPU fallback'),
    ('rift', dict(with_status=False), 'FAILED', 'without a runner status report'),
    ('radarsplat', dict(resume_file=False, with_status=False), 'TIMEOUT', 'radarsplat_gotcha.pt is missing'),
    ('spinr', dict(resume_file=False, with_status=False), 'TIMEOUT', 'before its first checkpoint'),
    ('rift', dict(resume_arg=lambda out: out/'checkpoint_best.pt'), 'TIMEOUT', 'already carries --resume'),
    ('sugavanam_ertin', dict(with_status=False), 'TIMEOUT', 'not implemented for sugavanam_ertin'),
    ('mfbp', dict(with_status=False), 'TIMEOUT', 'not implemented for mfbp'),
])
def test_gotcha_refusals(tmp_path, method, kwargs, state, message):
    root, task, _ = make_gotcha_root(tmp_path, method, **kwargs)
    with pytest.raises(RuntimeError, match=message):
        relaunch.classify(task, {JOB: dict(ROW, state=state)}, {}, root)


def test_gotcha_cancelled_after_start_is_still_refused(tmp_path):
    root, task, _ = make_gotcha_root(tmp_path, 'rift', with_status=False)
    with pytest.raises(RuntimeError, match='only SE Stage-1 checkpoint resume'):
        relaunch.classify(task, {JOB: dict(ROW, state='CANCELLED by 0')}, {}, root)


def test_gotcha_resume_of_a_resumed_run_keeps_the_command(tmp_path):
    root, task, out = make_gotcha_root(tmp_path, 'rift', directory='rift_nufft_full_native', with_status=False,
                                       resume_arg=lambda out: out/'checkpoint_latest.pt')
    kind, reason = relaunch.classify(task, {JOB: ROW}, {}, root)
    assert kind == 'resume-gotcha' and 'resubmitted unchanged' in reason


def test_gotcha_relaunch_appends_the_real_path_and_archives(tmp_path, monkeypatch, capsys):
    root, task, out = make_gotcha_root(tmp_path, 'rift', directory='rift_nufft_full_native', with_status=False)
    monkeypatch.setattr(relaunch, 'accounting', lambda ids: {JOB: ROW})
    monkeypatch.setattr(relaunch, 'live_queue', lambda: {})
    relaunch.relaunch(root, ('camry',), [], dry_run=True, keys=['camry-rift-full'])
    assert f'command += --resume {out}/checkpoint_latest.pt' in capsys.readouterr().out
    assert json.loads((root/'campaign.json').read_text())['tasks'][0]['job_id'] == JOB
    calls = []
    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=f'{2160000 + len(calls)}\n', stderr='')
    monkeypatch.setattr(relaunch.subprocess, 'run', fake_run)
    relaunch.relaunch(root, ('camry',), [], dry_run=False, keys=['camry-rift-full'])
    saved = json.loads((root/'campaign.json').read_text())['tasks'][0]
    assert saved['command'] == task['command'] + ['--resume', str(out/'checkpoint_latest.pt')]
    assert saved['command'].count('--resume') == 1 and 'auto' not in saved['command']
    assert saved['relaunch']['kind'] == 'resume-gotcha' and saved['relaunch']['previous_job_id'] == JOB
    assert saved['submission_history'][0]['command'] == task['command'] and saved['job_id'] == 2160001
    # A second wall-clock stop of the resumed job resubmits the same command unchanged.
    (root/'status'/f"camry-rift-full-{saved['job_id']}.json").write_text(json.dumps(dict(returncode=143, fallback=False)))
    monkeypatch.setattr(relaunch, 'accounting', lambda ids: {saved['job_id']: dict(ROW, state='FAILED')})
    relaunch.relaunch(root, ('camry',), [], dry_run=False, keys=['camry-rift-full'])
    again = json.loads((root/'campaign.json').read_text())['tasks'][0]
    assert again['command'] == saved['command'] and again['job_id'] == 2160002
    assert [h['job_id'] for h in again['submission_history']] == [JOB, 2160001]
    assert len(json.loads((root/'relaunches.json').read_text())) == 2
