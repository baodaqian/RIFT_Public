"""Move never-started tasks of a recorded PVC campaign to a campaign prepared from the current tree.

Relaunches through ``production_relaunch.py`` deliberately keep each campaign's saved
``source/`` snapshot, so a task still queued from an older snapshot never picks up later
working-tree changes (RIFT_PVC_Adaptation.md section 16: the C1 snapshot has no batched
lanes). This tool performs, as one recorded and resumable operation, the hand procedure
used for the Camry move to campaign C3:

1. The operator has run ``production_campaign.py prepare <new root> --scenes ... --methods
   ...`` from the current tree, which saves a fresh snapshot and metadata-only plans and
   leaves every task of the new root unsubmitted.
2. Each selected task of the old campaign must be PENDING, never started, and its output
   directory absent or empty. A Stage-2 task moves together with its Stage-1 parent (same
   batch, or a parent already superseded into the same new root).
3. Old jobs are cancelled first (Stage 2 before Stage 1) and must leave the queue; then the
   same task keys are submitted from the new root, Stage 1 before Stage 2 with ``afterok`` on
   the new parent and ``--kill-on-invalid-dep=yes``. No two jobs of one task ever run at once.
4. The old task gains ``superseded_by`` (campaign, job_id, cancelled_job_id, cancelled_at,
   reason), the field ``production_relaunch.classify`` refuses, and the new task gains
   ``supersedes``. ``supersessions.json`` at the new root keeps a batch ledger.

A task whose old job was cancelled by an earlier run that then failed before submitting
(``superseded_by.job_id`` is null and names this new root) is classified ``submit`` and
completed on the next run. ``--dry-run`` prints the plan and writes, cancels and submits
nothing. Nothing here edits ``production_relaunch.py`` or ``production_campaign.py``.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import production_relaunch as relaunch  # noqa: E402  (accounting, live_queue, sbatch_command, save)

DEFAULT_REASON = 'moved to a campaign prepared from the current working tree'
LIVE_STATES = ('RUNNING', 'PENDING', 'COMPLETING', 'REQUEUED', 'SUSPENDED', 'CONFIGURING')


def load_campaign(root):
    path = root/'campaign.json'
    if not path.is_file():
        raise RuntimeError(f'{root}: no campaign.json')
    campaign = json.loads(path.read_text())
    if campaign.get('schema') != 'rift_pvc_production_campaign_v1':
        raise RuntimeError(f'{root}: unexpected campaign schema {campaign.get("schema")!r}')
    return campaign


def output_is_empty(task):
    output = Path(task['output_dir'])
    return not output.exists() or not any(output.iterdir())


def classify(old, new, old_root, new_root, batch, account, queue):
    """Return (kind, reason) with kind in move/submit; raise on anything unexpected."""
    key = old['key']
    for field in ('scene', 'method', 'stage', 'depends_on'):
        if old.get(field) != new.get(field):
            raise RuntimeError(f'{key}: {field} differs between campaigns ({old.get(field)!r} vs {new.get(field)!r})')
    if Path(old['command'][2]).name != Path(new['command'][2]).name:
        raise RuntimeError(f'{key}: the new campaign runs a different script ({new["command"][2]})')
    if not str(new['command'][2]).startswith(str(new_root/'source')):
        raise RuntimeError(f'{key}: the new command does not run from {new_root/"source"}')
    if new.get('job_id') or new.get('submission_started'):
        raise RuntimeError(f'{key}: already submitted from {new_root} (job {new.get("job_id")})')
    if not output_is_empty(new):
        raise RuntimeError(f'{key}: new output directory {new["output_dir"]} is not empty')
    superseded = old.get('superseded_by')
    if superseded:
        if superseded.get('job_id') is None and superseded.get('campaign') == str(new_root):
            if old['job_id'] in queue:
                raise RuntimeError(f'{key}: recorded as cancelled but job {old["job_id"]} is still queued')
            return 'submit', (f'job {old["job_id"]} already cancelled at {superseded.get("cancelled_at")}; '
                              'submission from the new root is outstanding')
        raise RuntimeError(f'{key}: already superseded by {superseded}')
    job_id = old.get('job_id')
    if not job_id:
        raise RuntimeError(f'{key}: never submitted from {old_root}; nothing to supersede')
    if job_id not in queue:
        row = account.get(job_id)
        state = row['state'] if row else 'unknown to sacct'
        raise RuntimeError(f'{key}: job {job_id} is not in the live queue ({state}); '
                           'only queued, never-started tasks are moved')
    if queue[job_id]['state'] != 'PENDING':
        raise RuntimeError(f'{key}: job {job_id} is {queue[job_id]["state"]}, not PENDING')
    row = account.get(job_id) or {}
    if row.get('start') not in (None, 'None', 'Unknown', ''):
        raise RuntimeError(f'{key}: job {job_id} has a start time ({row["start"]}); refuse to move a started task')
    if old['stage'] != 'stage2' and not output_is_empty(old):
        raise RuntimeError(f'{key}: job {job_id} never started but {old["output_dir"]} is not empty')
    if old['stage'] == 'stage2':
        parent_key = old['depends_on']
        parent_superseded = None
        if parent_key not in batch:
            parent_superseded = batch.get('__all__', {}).get(parent_key, {}).get('superseded_by') or {}
            if parent_superseded.get('campaign') != str(new_root):
                raise RuntimeError(f'{key}: parent {parent_key} is neither in this batch nor superseded into {new_root}')
    return 'move', f'job {job_id} PENDING ({queue[job_id]["reason"]}), never started'


def plan_supersession(old_root, new_root, keys):
    old_campaign, new_campaign = load_campaign(old_root), load_campaign(new_root)
    if old_root == new_root:
        raise RuntimeError('old and new campaign roots are the same')
    if not (new_root/'source/scripts_pvc/production_job.sbatch').is_file():
        raise RuntimeError(f'{new_root}: no source snapshot with scripts_pvc/production_job.sbatch')
    old_tasks = {task['key']: task for task in old_campaign['tasks']}
    new_tasks = {task['key']: task for task in new_campaign['tasks']}
    unknown = sorted(set(keys) - set(old_tasks))
    if unknown:
        raise RuntimeError(f'unknown task keys in {old_root}: {unknown}')
    missing = sorted(set(keys) - set(new_tasks))
    if missing:
        raise RuntimeError(f'{new_root} has no plan for: {missing} (prepare it with --scenes/--methods covering them)')
    selected = [old_tasks[key] for key in keys]
    batch = {task['key']: task for task in selected}
    batch['__all__'] = old_tasks
    account = relaunch.accounting([task['job_id'] for task in selected if task.get('job_id')])
    queue = relaunch.live_queue()
    plan = []
    for old in sorted(selected, key=lambda task: task['stage'] == 'stage2'):
        kind, reason = classify(old, new_tasks[old['key']], old_root, new_root, batch, account, queue)
        plan.append(dict(old=old, new=new_tasks[old['key']], kind=kind, reason=reason))
    return old_campaign, new_campaign, new_tasks, plan, queue


def wait_until_gone(job_ids, timeout=90.0, interval=3.0):
    deadline = time.monotonic() + timeout
    while True:
        queue = relaunch.live_queue()
        remaining = [job_id for job_id in job_ids if job_id in queue and queue[job_id]['state'] in LIVE_STATES]
        if not remaining:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f'jobs {remaining} are still queued {timeout:.0f} s after scancel; '
                               'nothing was submitted; rerun once they are gone')
        time.sleep(interval)


def supersede(old_root, new_root, keys, reason, dry_run):
    old_path, new_path = old_root/'campaign.json', new_root/'campaign.json'
    old_campaign, new_campaign, new_tasks, plan, queue = plan_supersession(old_root, new_root, keys)
    now = datetime.now(timezone.utc).isoformat()
    for entry in plan:
        old, new = entry['old'], entry['new']
        parent = new_tasks[new['depends_on']] if new['depends_on'] else None
        dependency = None
        if parent is not None:
            dependency = f"afterok:{parent['job_id']}" if parent.get('job_id') else f"afterok:<new {parent['key']}>"
        entry['dependency'] = dependency
        print(f"{entry['kind']:>6} {old.get('job_id') or '-':>8} {old['key']:<36} dep={dependency or 'none':<28} {entry['reason']}",
              flush=True)
    if dry_run:
        print(f'dry run: {len(plan)} task(s) would move from {old_root} to {new_root}; nothing cancelled, submitted or written')
        return []
    # Phase A: cancel every old job that is still queued, dependents first, and wait for the queue to drop them.
    cancelled = []
    for entry in sorted(plan, key=lambda entry: entry['old']['stage'] != 'stage2'):
        old = entry['old']
        if entry['kind'] != 'move':
            continue
        result = subprocess.run(['scancel', str(old['job_id'])], text=True, capture_output=True, timeout=60)
        if result.returncode:
            raise RuntimeError(f"scancel {old['job_id']} failed for {old['key']}: {result.stderr}")
        old['superseded_by'] = dict(campaign=str(new_root), job_id=None, cancelled_job_id=old['job_id'],
                                    cancelled_at=datetime.now(timezone.utc).isoformat(), reason=reason)
        relaunch.save(old_path, old_campaign)
        cancelled.append(old['job_id'])
        print(f"cancelled {old['job_id']:>8} {old['key']}", flush=True)
    if cancelled:
        wait_until_gone(cancelled)
    # Phase B: submit from the new root, Stage 1 before Stage 2.
    ledger = []
    for entry in plan:
        old, new = entry['old'], entry['new']
        dependency = None
        if new['depends_on']:
            parent = new_tasks[new['depends_on']]
            if not parent.get('job_id'):
                raise RuntimeError(f"{new['key']}: parent {parent['key']} has no job in {new_root}")
            dependency = f"afterok:{parent['job_id']}"
        command = relaunch.sbatch_command(new_root, new, dependency)
        new['supersedes'] = dict(campaign=str(old_root), key=old['key'], cancelled_job_id=old['superseded_by']['cancelled_job_id'])
        if dependency:
            new['dependency'] = dependency
        new['submission_started'] = datetime.now(timezone.utc).isoformat()
        new['sbatch_command'] = command
        relaunch.save(new_path, new_campaign)
        result = subprocess.run(command, text=True, capture_output=True, timeout=90)
        new['submission_stdout'], new['submission_stderr'] = result.stdout, result.stderr
        if result.returncode:
            relaunch.save(new_path, new_campaign)
            raise RuntimeError(f"sbatch failed for {new['key']}: {result.stderr}\n"
                               f"{old['key']} stays recorded as cancelled with no new job; rerun this tool to submit it")
        new['job_id'] = int(result.stdout.strip().split(';')[0])
        relaunch.save(new_path, new_campaign)
        old['superseded_by']['job_id'] = new['job_id']
        relaunch.save(old_path, old_campaign)
        ledger.append(dict(key=new['key'], job_id=new['job_id'], dependency=dependency,
                           cancelled_job_id=old['superseded_by']['cancelled_job_id'], job_name=new['job_name']))
        print(f"submitted {new['job_id']:>8} {new['key']:<36} dep={dependency or 'none':<28} "
              f"(was {old['superseded_by']['cancelled_job_id']} in {old_root.name})", flush=True)
    ledger_path = new_root/'supersessions.json'
    record = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    record.append(dict(submitted_at=now, from_campaign=str(old_root), tasks=list(keys), reason=reason, jobs=ledger))
    relaunch.save(ledger_path, record)
    return ledger


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('old_root', type=Path, help='campaign root whose queued tasks move')
    parser.add_argument('new_root', type=Path, help='campaign root prepared from the current tree, not yet submitted')
    parser.add_argument('--tasks', nargs='+', required=True, help='task keys to move (a Stage-2 key needs its Stage-1 parent)')
    parser.add_argument('--reason', default=DEFAULT_REASON, help='recorded in superseded_by and supersessions.json')
    parser.add_argument('--dry-run', action='store_true', help='print the plan; cancel, submit and write nothing')
    args = parser.parse_args()
    if len(args.tasks) != len(set(args.tasks)):
        parser.error('task keys must be distinct')
    supersede(args.old_root.resolve(), args.new_root.resolve(), args.tasks, args.reason, args.dry_run)
