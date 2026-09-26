"""production_supersede.py moves queued, never-started tasks to a current-tree campaign root."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts_pvc'))
import production_relaunch as relaunch  # noqa: E402
import production_supersede as supersede  # noqa: E402


def make_root(tmp_path, name, job_ids):
    root = tmp_path/name
    (root/'source/scripts_pvc').mkdir(parents=True)
    (root/'source/scripts_pvc/production_job.sbatch').write_text('#!/bin/bash\n')
    (root/'logs').mkdir()
    tasks = []
    for key, stage, job_id in (('b787-sugavanam_ertin-stage1', 'stage1', job_ids[0]),
                               ('b787-sugavanam_ertin-stage2', 'stage2', job_ids[1]),
                               ('b787-rift-full', 'full', job_ids[2])):
        method = 'rift' if key.endswith('rift-full') else 'sugavanam_ertin'
        tasks.append(dict(key=key, scene='b787', method=method, stage=stage,
                          output_dir=str(root/'outputs'/method),
                          command=['python', '-u', str(root/'source/train_rift_dataset_pvc.py'), '--method', method],
                          depends_on='b787-sugavanam_ertin-stage1' if stage == 'stage2' else None,
                          job_name='pvcprod-'+key.replace('sugavanam_ertin', 'se'), job_id=job_id))
    campaign = dict(schema='rift_pvc_production_campaign_v1', root=str(root), tasks=tasks)
    relaunch.save(root/'campaign.json', campaign)
    return root


@pytest.fixture
def slurm(monkeypatch):
    """Fake squeue/sacct/scancel/sbatch; the queue drops a job when it is cancelled."""
    state = dict(queue={111: dict(state='PENDING', reason='QOSMaxGRESPerUser'),
                        112: dict(state='PENDING', reason='Dependency'),
                        113: dict(state='PENDING', reason='QOSMaxGRESPerUser')},
                 account={111: dict(state='PENDING', start='None', end='Unknown', elapsed='00:00:00', nodes='None assigned'),
                          112: dict(state='PENDING', start='None', end='Unknown', elapsed='00:00:00', nodes='None assigned'),
                          113: dict(state='PENDING', start='None', end='Unknown', elapsed='00:00:00', nodes='None assigned')},
                 calls=[], next_job=2001)
    monkeypatch.setattr(relaunch, 'live_queue', lambda: dict(state['queue']))
    monkeypatch.setattr(relaunch, 'accounting', lambda ids: {i: state['account'][i] for i in ids if i in state['account']})

    class Result:
        def __init__(self, stdout, returncode=0):
            self.stdout, self.stderr, self.returncode = stdout, '', returncode

    def run(command, **kwargs):
        state['calls'].append(list(command))
        if command[0] == 'scancel':
            state['queue'].pop(int(command[1]), None)
            return Result('')
        assert command[0] == 'sbatch' and command[1] == '--parsable'
        job = state['next_job']
        state['next_job'] += 1
        return Result(f'{job}\n')
    monkeypatch.setattr(supersede.subprocess, 'run', run)
    monkeypatch.setattr(supersede.time, 'sleep', lambda s: None)
    return state


def test_dry_run_plans_without_touching_anything(tmp_path, slurm, capsys):
    old = make_root(tmp_path, 'old', (111, 112, 113))
    new = make_root(tmp_path, 'new', (None, None, None))
    before = (old/'campaign.json').read_text(), (new/'campaign.json').read_text()
    assert supersede.supersede(old, new, ['b787-sugavanam_ertin-stage1', 'b787-sugavanam_ertin-stage2'], 'test', dry_run=True) == []
    assert slurm['calls'] == []
    assert ((old/'campaign.json').read_text(), (new/'campaign.json').read_text()) == before
    out = capsys.readouterr().out
    assert 'b787-sugavanam_ertin-stage1' in out and 'afterok:<new b787-sugavanam_ertin-stage1>' in out and 'dry run' in out


def test_move_cancels_dependents_first_then_submits_from_the_new_root(tmp_path, slurm):
    old = make_root(tmp_path, 'old', (111, 112, 113))
    new = make_root(tmp_path, 'new', (None, None, None))
    keys = ['b787-sugavanam_ertin-stage2', 'b787-sugavanam_ertin-stage1']
    ledger = supersede.supersede(old, new, keys, 'batched lanes', dry_run=False)
    scancels = [c for c in slurm['calls'] if c[0] == 'scancel']
    sbatches = [c for c in slurm['calls'] if c[0] == 'sbatch']
    assert scancels == [['scancel', '112'], ['scancel', '111']]
    assert [c[-1] for c in sbatches] == ['b787-sugavanam_ertin-stage1', 'b787-sugavanam_ertin-stage2']
    assert all(c[-2] == str(new) and c[-3] == str(new/'source/scripts_pvc/production_job.sbatch') for c in sbatches)
    assert '--dependency' in sbatches[1] and sbatches[1][sbatches[1].index('--dependency')+1] == 'afterok:2001'
    assert '--kill-on-invalid-dep=yes' in sbatches[1] and '--dependency' not in sbatches[0]
    assert slurm['calls'].index(scancels[-1]) < slurm['calls'].index(sbatches[0])
    old_tasks = {t['key']: t for t in json.loads((old/'campaign.json').read_text())['tasks']}
    new_tasks = {t['key']: t for t in json.loads((new/'campaign.json').read_text())['tasks']}
    stage1, stage2 = old_tasks['b787-sugavanam_ertin-stage1'], old_tasks['b787-sugavanam_ertin-stage2']
    assert stage1['job_id'] == 111 and stage2['job_id'] == 112            # history kept; the guard field is added
    assert set(stage1['superseded_by']) == {'campaign', 'job_id', 'cancelled_job_id', 'cancelled_at', 'reason'}
    assert stage1['superseded_by']['campaign'] == str(new) and stage1['superseded_by']['job_id'] == 2001
    assert stage1['superseded_by']['cancelled_job_id'] == 111 and stage1['superseded_by']['reason'] == 'batched lanes'
    assert stage2['superseded_by']['job_id'] == 2002 and stage2['superseded_by']['cancelled_job_id'] == 112
    assert 'superseded_by' not in old_tasks['b787-rift-full']
    assert new_tasks['b787-sugavanam_ertin-stage1']['job_id'] == 2001
    assert new_tasks['b787-sugavanam_ertin-stage2']['job_id'] == 2002
    assert new_tasks['b787-sugavanam_ertin-stage2']['dependency'] == 'afterok:2001'
    assert new_tasks['b787-sugavanam_ertin-stage1']['supersedes'] == dict(campaign=str(old), key='b787-sugavanam_ertin-stage1', cancelled_job_id=111)
    assert new_tasks['b787-rift-full']['job_id'] is None
    record = json.loads((new/'supersessions.json').read_text())
    assert len(record) == 1 and [j['job_id'] for j in record[0]['jobs']] == [2001, 2002] and record[0]['from_campaign'] == str(old)
    assert [j['job_id'] for j in ledger] == [2001, 2002]
    # The relaunch tool now refuses the superseded tasks.
    with pytest.raises(RuntimeError, match='superseded by'):
        relaunch.classify(stage1, slurm['account'], slurm['queue'])


def test_refusals_leave_records_untouched(tmp_path, slurm):
    old = make_root(tmp_path, 'old', (111, 112, 113))
    new = make_root(tmp_path, 'new', (None, None, None))
    before = (old/'campaign.json').read_text()
    # A running job is never moved.
    slurm['queue'][113] = dict(state='RUNNING', reason='None')
    with pytest.raises(RuntimeError, match='RUNNING, not PENDING'):
        supersede.supersede(old, new, ['b787-rift-full'], 'x', dry_run=False)
    # A job that already started (sacct start time) is never moved, even if requeued as PENDING.
    slurm['queue'][113] = dict(state='PENDING', reason='Priority')
    slurm['account'][113]['start'] = '2026-09-22T10:00:00'
    with pytest.raises(RuntimeError, match='start time'):
        supersede.supersede(old, new, ['b787-rift-full'], 'x', dry_run=False)
    slurm['account'][113]['start'] = 'None'
    # A non-empty output directory means the task is not a fresh move.
    out = Path(json.loads(before)['tasks'][2]['output_dir'])
    out.mkdir(parents=True)
    (out/'checkpoint_latest.pt').write_bytes(b'x')
    with pytest.raises(RuntimeError, match='not empty'):
        supersede.supersede(old, new, ['b787-rift-full'], 'x', dry_run=False)
    (out/'checkpoint_latest.pt').unlink()
    # A Stage-2 task needs its parent in the batch (or already superseded into the same root).
    with pytest.raises(RuntimeError, match='parent'):
        supersede.supersede(old, new, ['b787-sugavanam_ertin-stage2'], 'x', dry_run=False)
    # A task the new root already submitted is refused.
    campaign = json.loads((new/'campaign.json').read_text())
    campaign['tasks'][2]['job_id'] = 999
    relaunch.save(new/'campaign.json', campaign)
    with pytest.raises(RuntimeError, match='already submitted'):
        supersede.supersede(old, new, ['b787-rift-full'], 'x', dry_run=False)
    # A job gone from the queue (finished, cancelled by hand) is not this tool's business.
    campaign['tasks'][2]['job_id'] = None
    relaunch.save(new/'campaign.json', campaign)
    slurm['queue'].pop(113)
    with pytest.raises(RuntimeError, match='not in the live queue'):
        supersede.supersede(old, new, ['b787-rift-full'], 'x', dry_run=False)
    assert (old/'campaign.json').read_text() == before
    assert slurm['calls'] == []


def test_interrupted_move_resumes_with_submit_only(tmp_path, slurm, monkeypatch):
    old = make_root(tmp_path, 'old', (111, 112, 113))
    new = make_root(tmp_path, 'new', (None, None, None))
    real_run = supersede.subprocess.run

    def failing_sbatch(command, **kwargs):
        if command[0] == 'sbatch':
            class Result:
                stdout, stderr, returncode = '', 'sbatch: error: QOS limit', 1
            slurm['calls'].append(list(command))
            return Result()
        return real_run(command, **kwargs)
    monkeypatch.setattr(supersede.subprocess, 'run', failing_sbatch)
    with pytest.raises(RuntimeError, match='sbatch failed'):
        supersede.supersede(old, new, ['b787-rift-full'], 'x', dry_run=False)
    old_task = json.loads((old/'campaign.json').read_text())['tasks'][2]
    assert old_task['superseded_by']['job_id'] is None and old_task['superseded_by']['cancelled_job_id'] == 113
    with pytest.raises(RuntimeError, match='superseded by'):
        relaunch.classify(old_task, slurm['account'], slurm['queue'])
    new_task = json.loads((new/'campaign.json').read_text())['tasks'][2]
    assert new_task['job_id'] is None and new_task['submission_started']
    # The new task's failed submission is cleared by the operator's inspection; here the retry is the tool itself.
    new_campaign = json.loads((new/'campaign.json').read_text())
    for field in ('submission_started', 'sbatch_command', 'submission_stdout', 'submission_stderr', 'supersedes'):
        new_campaign['tasks'][2].pop(field, None)
    relaunch.save(new/'campaign.json', new_campaign)
    monkeypatch.setattr(supersede.subprocess, 'run', real_run)
    ledger = supersede.supersede(old, new, ['b787-rift-full'], 'x', dry_run=False)
    assert [j['job_id'] for j in ledger] == [2001]
    assert [c for c in slurm['calls'] if c[0] == 'scancel'] == [['scancel', '113']]
    old_task = json.loads((old/'campaign.json').read_text())['tasks'][2]
    assert old_task['superseded_by']['job_id'] == 2001
