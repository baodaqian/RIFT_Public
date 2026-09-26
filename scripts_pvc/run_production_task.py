"""Execute one recorded task, retaining child failures and rejecting XPU fallback."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from datetime import datetime, timezone


def main():
    root, task = Path(sys.argv[1]), sys.argv[2]
    manifest = json.loads((root/'campaign.json').read_text())
    entry = next(e for e in manifest['tasks'] if e['key'] == task)
    import torch
    from rift_pvc import accelerator
    assert accelerator.backend() == 'xpu' and torch.xpu.device_count() == 1
    print(json.dumps({'task': task, 'job_id': os.environ['SLURM_JOB_ID'],
                      'accelerator': accelerator.describe(), 'torch': torch.__version__,
                      'command': entry['command']}, default=str), flush=True)
    log = root/'logs'/f'{task}-{os.environ["SLURM_JOB_ID"]}.trainer.log'
    fallback = False
    with log.open('w', buffering=1) as stream:
        child = subprocess.Popen(entry['command'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, bufsize=1, start_new_session=True)
        def stop(signum, frame):
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        for line in child.stdout:
            stream.write(line)
            print(line, end='', flush=True)
            if 'Aten Op fallback from XPU to CPU' in line:
                fallback = True
                stop(signal.SIGTERM, None)
        rc = child.wait()
    if fallback:
        rc = 86
    if rc == 0 and entry['stage'] == 'stage1':
        # Only the converged workflow publishes this file. A zero shell status
        # without the stage handoff must never release the dependent job.
        if not (Path(entry['output_dir'])/'checkpoint_stage1_final.pt').is_file():
            rc = 87
    report = dict(task=task, job_id=os.environ['SLURM_JOB_ID'], returncode=rc,
                  fallback=fallback, finished_at=datetime.now(timezone.utc).isoformat())
    (root/'status'/f'{task}-{os.environ["SLURM_JOB_ID"]}.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)
    return rc if rc >= 0 else 128-rc


if __name__ == '__main__':
    raise SystemExit(main())
