"""Relaunch cancelled tasks of a recorded PVC production campaign.

Two kinds of relaunch, decided from Slurm accounting and the task's artifacts:

* ``fresh`` -- the recorded job was cancelled before it started and the task's
  output directory is still empty, so the recorded command is submitted again
  unchanged.
* ``resume`` -- the recorded Sugavanam-Ertin Stage-1 job started, was cancelled
  and left a durable ``checkpoint_latest.pt`` in phase ``stage1``; the recorded
  command is resubmitted with ``--resume <that checkpoint>`` appended, keeping
  ``--se-stage1-only``, the original output root and the campaign's saved
  source snapshot.

* ``resume-auto`` -- a GeRaF task on the collection frontend whose job hit its
  time limit (``TIMEOUT``) or exited on the SIGTERM warning (``FAILED`` with
  runner status 143) and left ``checkpoints/checkpoint_latest.pth.tar``; the
  recorded command is resubmitted with ``--resume auto`` (the planner flag that
  admits the existing output; GeRaF itself reloads its latest checkpoint).
* ``resume-gotcha`` -- a task on the GOTCHA frontend (``train_gotcha_dataset_pvc.py``)
  stopped the same way (the same gate: ``TIMEOUT``, or ``FAILED`` with runner status
  143 and no XPU fallback; ``FAILED`` without a status report is refused). The
  frontend's ``--resume`` takes the real file each backend resumes from, inside the
  task's method output directory (``GOTCHA_RESUME_FILES``); it must exist, and it is
  appended as that path. A command that already resumes exactly that file (an
  earlier ``resume-gotcha``) is resubmitted unchanged; any other ``--resume`` is
  refused. Each trainer checks the checkpoint's dataset and recipe identity before
  reading any response. SE (stage semantics) and MFBP (nothing to resume) are refused.

Running or queued tasks are never touched. Stage-2 tasks are resubmitted with
``afterok`` on their parent's *new* job ID. Every earlier submission is kept
under ``submission_history`` in ``campaign.json``; the top-level ``job_id`` is
always the latest submission. ``--dry-run`` prints the plan and writes nothing.
"""
import argparse
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import subprocess
import sys

RIFT_SCENES = ('a320', 'x59', 'firetruck', 'racecar', 'loader', 'b787')
GOTCHA_FRONTEND = 'train_gotcha_dataset_pvc.py'
# The file each GOTCHA backend's --resume must name, inside the task's method output directory:
# the built-in RIFT trainers (rift_pvc/gotcha_training.py), SpINR (rift_pvc/spinr_gotcha_training.py
# accepts only this name in its own directory), Radar Fields (rift_pvc/radar_fields_gotcha.py), GeRaF
# (rift_pvc/geraf_source_training.py; not the collection lane's checkpoints/ subdirectory) and
# RadarSplat (its run-level control file rift/radarsplat_gotcha.py CONTROL_FILE, written at start;
# heads and the target cache resume through it).
GOTCHA_RESUME_FILES = {'rift': 'checkpoint_latest.pt', 'rift_grid': 'checkpoint_latest.pt',
                       'isotropic': 'checkpoint_latest.pt', 'spinr': 'checkpoint_latest.pt',
                       'radar_fields': 'checkpoint_latest.pt', 'geraf': 'checkpoint_latest.pth.tar',
                       'radarsplat': 'radarsplat_gotcha.pt'}
SUBMISSION_FIELDS = ('job_id', 'dependency', 'submission_started', 'sbatch_command',
                     'submission_stdout', 'submission_stderr')


def save(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def accounting(job_ids):
    """State, start and elapsed of each recorded job ID from sacct."""
    if not job_ids:
        return {}
    out = subprocess.check_output(['sacct', '-X', '-P', '-n', '-j', ','.join(map(str, job_ids)),
                                   '-o', 'JobID,State,Start,End,Elapsed,NodeList'], text=True)
    rows = {}
    for line in out.splitlines():
        if not line.strip():
            continue
        job_id, state, start, end, elapsed, nodes = line.split('|')[:6]
        rows[int(job_id)] = dict(state=state, start=start, end=end, elapsed=elapsed, nodes=nodes)
    return rows


def live_queue():
    """The current user's queue; unknown IDs are looked up by intersection, never by -j."""
    out = subprocess.check_output(['squeue', '-h', '-u', getpass.getuser(), '-o', '%i|%T|%r'], text=True)
    queue = {}
    for line in out.splitlines():
        if line.strip():
            job_id, state, reason = line.split('|')[:3]
            queue[int(job_id)] = dict(state=state, reason=reason)
    return queue


def classify(task, account, queue, root=None):
    """Return (kind, reason) for one task; kind in fresh/resume/skip; raise on anything unexpected."""
    job_id = task.get('job_id')
    if job_id is None:
        raise RuntimeError(f"{task['key']}: never submitted; this tool relaunches cancelled tasks only")
    if task.get('superseded_by'):
        raise RuntimeError(f"{task['key']}: superseded by {task['superseded_by']}; relaunch it there, not here")
    if task.get('submission_started') and not job_id:
        raise RuntimeError(f"{task['key']}: unresolved submission; inspect Slurm before retrying")
    if job_id in queue:
        return 'skip', f"live in queue ({queue[job_id]['state']} {queue[job_id]['reason']})"
    row = account.get(job_id)
    if row is None:
        raise RuntimeError(f"{task['key']}: job {job_id} unknown to sacct")
    if row['state'] in ('RUNNING', 'PENDING', 'COMPLETING', 'REQUEUED', 'SUSPENDED'):
        return 'skip', f"live according to sacct ({row['state']})"
    if row['state'] in ('TIMEOUT', 'FAILED') and frontend(task) == GOTCHA_FRONTEND:
        return classify_gotcha_interrupted(task, row, root)
    if row['state'] in ('TIMEOUT', 'FAILED') and task['method'] == 'geraf':
        return classify_geraf_interrupted(task, row, root)
    if not row['state'].startswith('CANCELLED'):
        raise RuntimeError(f"{task['key']}: job {job_id} is {row['state']}, not cancelled; refuse to relaunch")
    output = Path(task['output_dir'])
    started = row['start'] not in ('None', 'Unknown', '')
    if not started:
        if task['stage'] != 'stage2' and output.exists() and any(output.iterdir()):
            raise RuntimeError(f"{task['key']}: job {job_id} never started but {output} is not empty")
        if task['stage'] == 'stage2' and '--resume' not in task['command']:
            raise RuntimeError(f"{task['key']}: stage2 command lacks --resume")
        return 'fresh', f"job {job_id} cancelled before start ({row['end']})"
    if task['method'] != 'sugavanam_ertin' or task['stage'] != 'stage1':
        raise RuntimeError(f"{task['key']}: job {job_id} ran {row['elapsed']} before cancellation; "
                           "only SE Stage-1 checkpoint resume is implemented here")
    if '--resume' in task['command']:
        raise RuntimeError(f"{task['key']}: command already carries --resume")
    checkpoint = output/'checkpoint_latest.pt'
    status_path = output/'status.json'
    if not checkpoint.is_file() or not status_path.is_file():
        raise RuntimeError(f"{task['key']}: no durable checkpoint_latest.pt/status.json under {output}")
    status = json.loads(status_path.read_text())
    if status.get('phase') != 'stage1':
        raise RuntimeError(f"{task['key']}: checkpoint phase is {status.get('phase')!r}, not stage1")
    if (output/'checkpoint_stage1_final.pt').exists():
        raise RuntimeError(f"{task['key']}: Stage 1 already finished; nothing to resume")
    return 'resume', (f"job {job_id} ran {row['elapsed']} on {row['nodes']}, cancelled {row['end']}; "
                      f"checkpoint at iteration {status.get('iteration')} cursor {status.get('group_cursor')}")


def geraf_checkpoint(task):
    return Path(task['output_dir'])/'checkpoints'/'checkpoint_latest.pth.tar'


def frontend(task):
    return next((Path(a).name for a in task['command'] if a.endswith('.py')), None)


def gotcha_resume_file(task):
    return Path(task['output_dir'])/GOTCHA_RESUME_FILES[task['method']]


def check_clean_interruption(task, row, root):
    """The wall-clock gate: TIMEOUT, or a runner status report of exactly 143 without XPU fallback."""
    job_id = task['job_id']
    status_path = Path(root)/'status'/f"{task['key']}-{job_id}.json" if root else None
    if status_path is not None and status_path.is_file():
        status = json.loads(status_path.read_text())
        if status.get('fallback'):
            raise RuntimeError(f"{task['key']}: job {job_id} stopped on an XPU fallback; not a wall-clock interruption")
        if status.get('returncode') != 143:
            raise RuntimeError(f"{task['key']}: job {job_id} runner exit {status.get('returncode')}, not the clean SIGTERM interruption (143)")
    elif row['state'] != 'TIMEOUT':
        raise RuntimeError(f"{task['key']}: job {job_id} FAILED without a runner status report; inspect the log before resuming")


def classify_gotcha_interrupted(task, row, root):
    """A GOTCHA-frontend task stopped by the wall clock: resume its backend's own file, or refuse."""
    job_id, method = task['job_id'], task['method']
    if method not in GOTCHA_RESUME_FILES:
        raise RuntimeError(f"{task['key']}: GOTCHA wall-clock resume is not implemented for {method}")
    checkpoint = gotcha_resume_file(task)
    command = task['command']
    if '--resume' in command:
        at = command.index('--resume')
        given = command[at + 1] if at + 1 < len(command) else None
        if command.count('--resume') != 1 or given is None or Path(given) != checkpoint:
            raise RuntimeError(f"{task['key']}: command already carries --resume {given}, not this run's {checkpoint}")
    if not checkpoint.is_file():
        raise RuntimeError(f"{task['key']}: job {job_id} is {row['state']} but {checkpoint} is missing "
                           "(stopped before its first checkpoint?); inspect the log")
    check_clean_interruption(task, row, root)
    again = ' (command already resumes it; resubmitted unchanged)' if '--resume' in command else ''
    return 'resume-gotcha', (f"job {job_id} {row['state']} after {row['elapsed']} on {row['nodes']} ({row['end']}); "
                             f"{method} resumes {checkpoint.name}{again}")


def classify_geraf_interrupted(task, row, root):
    """GeRaF stopped by the wall clock: resume its own latest checkpoint, or refuse."""
    job_id = task['job_id']
    if '--resume' in task['command']:
        raise RuntimeError(f"{task['key']}: command already carries --resume")
    script = next((Path(a).name for a in task['command'] if a.endswith('.py')), None)
    if script != 'train_rift_dataset_pvc.py':
        raise RuntimeError(f"{task['key']}: GeRaF wall-clock resume is implemented for the collection frontend only, not {script}")
    checkpoint = geraf_checkpoint(task)
    if not checkpoint.is_file():
        raise RuntimeError(f"{task['key']}: job {job_id} is {row['state']} but {checkpoint} is missing")
    check_clean_interruption(task, row, root)
    return 'resume-auto', (f"job {job_id} {row['state']} after {row['elapsed']} on {row['nodes']} ({row['end']}); "
                           f"GeRaF resumes {checkpoint.name}")


def plan_relaunch(root, scenes, keys=None):
    campaign = json.loads((root/'campaign.json').read_text())
    tasks = {task['key']: task for task in campaign['tasks']}
    if keys:
        unknown = sorted(set(keys) - set(tasks))
        if unknown:
            raise RuntimeError(f'unknown task keys: {unknown}')
    selected = [task for task in campaign['tasks']
                if task['scene'] in scenes and (not keys or task['key'] in keys)]
    account = accounting([task['job_id'] for task in selected if task.get('job_id')])
    queue = live_queue()
    plan = []
    for task in sorted(selected, key=lambda task: task['stage'] == 'stage2'):
        kind, reason = classify(task, account, queue, root)
        if kind == 'skip':
            plan.append(dict(task=task, kind=kind, reason=reason))
            continue
        if task['depends_on']:
            # The parent is either relaunched in this same run or already live from an earlier batch.
            parent = tasks[task['depends_on']]
            planned = next((p for p in plan if p['task'] is parent), None)
            if (planned is None or planned['kind'] == 'skip') and parent.get('job_id') not in queue:
                raise RuntimeError(f"{task['key']}: parent {parent['key']} is neither relaunched now nor live in the queue")
        entry = dict(task=task, kind=kind, reason=reason)
        if kind == 'resume':
            entry['resume_checkpoint'] = str(Path(task['output_dir'])/'checkpoint_latest.pt')
        elif kind == 'resume-auto':
            entry['resume_checkpoint'] = str(geraf_checkpoint(task))
        elif kind == 'resume-gotcha':
            entry['resume_checkpoint'] = str(gotcha_resume_file(task))
        plan.append(entry)
    return campaign, tasks, plan, queue


def sbatch_command(root, task, dependency):
    command = ['sbatch', '--parsable', '--export=ALL', '--job-name', task['job_name'],
               '--output', str(root/'logs'/f"{task['key']}-%j.slurm.log"),
               '--chdir', str(root/'source')]
    if dependency:
        command += ['--dependency', dependency, '--kill-on-invalid-dep=yes']
    command += [str(root/'source/scripts_pvc/production_job.sbatch'), str(root), task['key']]
    return command


def relaunch(root, scenes, after, dry_run, keys=None):
    path = root/'campaign.json'
    campaign, tasks, plan, queue = plan_relaunch(root, scenes, keys)
    gate = [job_id for job_id in after if job_id in queue]
    dropped = [job_id for job_id in after if job_id not in queue]
    if dropped:
        print(f"note: gate jobs {dropped} are no longer queued; no dependency on them", flush=True)
    gate_dependency = 'afterany:' + ':'.join(map(str, gate)) if gate else None
    now = datetime.now(timezone.utc).isoformat()
    ledger = []
    for entry in plan:
        task, kind = entry['task'], entry['kind']
        if kind == 'skip':
            print(f"   skip {task['job_id']:>8} {task['key']:<36} {entry['reason']}", flush=True)
            continue
        if task['depends_on']:
            parent = tasks[task['depends_on']]
            dependency = f"afterok:{parent['job_id']}" if not dry_run else f"afterok:<new {parent['key']}>"
        else:
            dependency = gate_dependency
        command = list(task['command'])
        if kind == 'resume':
            command += ['--resume', entry['resume_checkpoint']]
        elif kind == 'resume-auto':
            command += ['--resume', 'auto']
        elif kind == 'resume-gotcha' and '--resume' not in command:
            command += ['--resume', entry['resume_checkpoint']]
        sbatch = sbatch_command(root, task, dependency)
        if dry_run:
            print(f"{kind:>7} {task['job_id']:>8} {task['key']:<36} dep={dependency or 'none':<28} {entry['reason']}")
            if kind == 'resume':
                print(f"         command += --resume {entry['resume_checkpoint']}")
            elif kind == 'resume-auto':
                print(f"         command += --resume auto  (reloads {entry['resume_checkpoint']})")
            elif kind == 'resume-gotcha':
                print(f"         command {'already resumes' if '--resume' in task['command'] else '+= --resume'} "
                      f"{entry['resume_checkpoint']}")
            continue
        history = task.setdefault('submission_history', [])
        archived = {field: task.pop(field) for field in SUBMISSION_FIELDS if field in task}
        archived.update(outcome=entry['reason'], archived_at=now)
        if kind in ('resume', 'resume-auto', 'resume-gotcha'):
            archived['command'] = task['command']
            task['command'] = command
        history.append(archived)
        task['relaunch'] = dict(kind=kind, previous_job_id=archived.get('job_id'), reason=entry['reason'],
                                resume_checkpoint=entry.get('resume_checkpoint'), gate=gate)
        task['job_id'] = None
        if dependency:
            task['dependency'] = dependency
        task['submission_started'] = datetime.now(timezone.utc).isoformat()
        task['sbatch_command'] = sbatch
        save(path, campaign)
        result = subprocess.run(sbatch, text=True, capture_output=True, timeout=90)
        task['submission_stdout'], task['submission_stderr'] = result.stdout, result.stderr
        if result.returncode:
            save(path, campaign)
            raise RuntimeError(f"sbatch failed for {task['key']}: {result.stderr}")
        task['job_id'] = int(result.stdout.strip().split(';')[0])
        save(path, campaign)
        ledger.append(dict(key=task['key'], kind=kind, job_id=task['job_id'], dependency=dependency,
                           previous_job_id=archived.get('job_id'), job_name=task['job_name']))
        print(f"{kind:>7} {task['job_id']:>8} {task['key']:<36} dep={dependency or 'none':<28} "
              f"(was {archived.get('job_id')})", flush=True)
    if not dry_run and ledger:
        ledger_path = root/'relaunches.json'
        record = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
        record.append(dict(submitted_at=now, scenes=list(scenes), tasks=list(keys or []), gate=gate, jobs=ledger))
        save(ledger_path, record)
    return ledger


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('root', type=Path, help='campaign root holding campaign.json and source/')
    parser.add_argument('--scenes', nargs='+', default=list(RIFT_SCENES),
                        help='scenes to relaunch (default: the six RIFT-dataset scenes; Camry stays deferred)')
    parser.add_argument('--after', type=int, nargs='*', default=[],
                        help='job IDs every first-stage relaunch must wait for (afterany), if still queued')
    parser.add_argument('--tasks', nargs='+', help='relaunch only these task keys (a batch); default: every cancelled task of the scenes')
    parser.add_argument('--dry-run', action='store_true', help='print the plan; submit and write nothing')
    args = parser.parse_args()
    relaunch(args.root.resolve(), tuple(args.scenes), args.after, args.dry_run, args.tasks)
