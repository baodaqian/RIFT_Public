"""Prepare and submit selected seven-scene PVC production campaigns.

Source provenance uses repository URLs/commits and a file inventory, never
content hashes. Preparation performs canonical metadata-only planning; Slurm
execution owns all response access, caches and checkpoints.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCENES = ('a320', 'x59', 'firetruck', 'racecar', 'loader', 'b787', 'camry')
METHODS = ('rift', 'geraf', 'spinr', 'sugavanam_ertin')
ALL_METHODS = (*METHODS, 'radar_fields', 'radarsplat')
# Extra Camry adaptive-RIFT flags per variant; 'default' adds none, so default plans are unchanged.
GOTCHA_RIFT_VARIANTS = {'default': [],
                        'nufft_full_native': ['--forward-evaluation', 'nufft', '--loss-domain', 'full_native']}


def save(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def prepare(root, methods=METHODS, scenes=SCENES, gotcha_rift_point_chunk=16384, gotcha_rift_variant='default'):
    root.mkdir(parents=True, exist_ok=False)
    for name in ('source', 'plans', 'logs', 'status', 'runtime'):
        (root/name).mkdir()
    paths = sorted(set(git('ls-files', '-z', '--cached', '--others', '--exclude-standard').split('\0')))
    inventory = []
    for relative in paths:
        source = ROOT/relative
        if not relative or not source.is_file():
            continue
        destination = root/'source'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        inventory.append(relative)
    provenance = dict(repository=git('remote', 'get-url', 'origin'), commit=git('rev-parse', 'HEAD'),
                      worktree_status=git('status', '--short'), files=inventory,
                      source_snapshot=str(root/'source'),
                      geraf_upstream='https://github.com/VictorLlu/GeRaF-SENS',
                      geraf_commit='38266cb6e194e2f3dcbead614069a7281ffd21a5',
                      created_at=datetime.now(timezone.utc).isoformat())
    references = []
    if 'radar_fields' in methods:
        relative = 'external/RadarFields_reference'
        if (ROOT/relative/'.git').exists():
            references.append(dict(path=relative,
                repository=subprocess.check_output(['git', '-C', str(ROOT/relative), 'remote', 'get-url', 'origin'], text=True).strip(),
                commit=subprocess.check_output(['git', '-C', str(ROOT/relative), 'rev-parse', 'HEAD'], text=True).strip()))
        else:
            # Public snapshots include the source files without a nested Git
            # repository. Querying git in that directory would otherwise record
            # the enclosing RIFT repository as Radar Fields' upstream.
            packaged = json.loads((ROOT/'SOURCE_PROVENANCE.json').read_text())
            reference = next(item for item in packaged['external_sources']
                             if item['path'] == relative)
            references.append({key: reference[key]
                               for key in ('path', 'repository', 'commit')})
    if 'radarsplat' in methods:
        contract = json.loads((ROOT/'protocols/radarsplat_official_reference.json').read_text())
        references.append(dict(path='external/radarsplat_reference/'+contract['commit'],
                               repository=contract['repository'], commit=contract['commit']))
    for reference in references:
        relative = reference['path']
        shutil.copytree(ROOT/relative, root/'source'/relative,
                        ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc'), dirs_exist_ok=True)
    provenance['external_sources'] = references
    save(root/'source_provenance.json', provenance)
    env = dict(os.environ, PYTHONPATH=str(root/'source'))
    plan_command = [sys.executable, str(root/'source/scripts_pvc/production_campaign.py'),
                    '_plan', str(root), '--methods', *methods, '--scenes', *scenes]
    if gotcha_rift_point_chunk:
        plan_command += ['--gotcha-rift-point-chunk', str(gotcha_rift_point_chunk)]
    if gotcha_rift_variant != 'default':
        plan_command += ['--gotcha-rift-variant', gotcha_rift_variant]
    subprocess.run(plan_command, cwd=root/'source', env=env, check=True)


def plan(root, methods=METHODS, scenes=SCENES, gotcha_rift_point_chunk=16384, gotcha_rift_variant='default'):
    sys.path.insert(0, str(ROOT))
    import train_rift_dataset_pvc as collection
    import train_gotcha_dataset_pvc as gotcha
    from rift.rift_dataset import role_manifest
    tasks = []
    for scene in scenes:
        for method in methods:
            stages = ('stage1', 'stage2') if method == 'sugavanam_ertin' else ('full',)
            previous = None
            for stage in stages:
                key = f'{scene}-{method}-{stage}'
                if scene == 'camry':
                    args = ['--dataset-root', os.environ['GOTCHA_DATA_ROOT'],
                            '--output-root', str(root/'outputs/gotcha'), '--region', 'camry',
                            '--polarizations', 'hh', '--passes', *map(str, range(1, 9)),
                            '--num-train', '1500', '--num-tx', '1', '--num-rx', '1',
                            '--pulses-per-sector', '16', '--frequency-stride', '2',
                            '--method', method, '--device', 'xpu']
                    if method == 'rift' and gotcha_rift_point_chunk:
                        # Pinned GOTCHA recipe (2026-09-22): batched lane with --point-chunk 16384.
                        args += ['--point-chunk', str(gotcha_rift_point_chunk)]
                    if method == 'rift':
                        # Opt-in control (alignment doc section 15); the default adds nothing.
                        args += GOTCHA_RIFT_VARIANTS[gotcha_rift_variant]
                    if method == 'spinr':
                        args += ['--method-config', str(ROOT/'protocols/gotcha_spinr_g48_midpoint.json')]
                    elif method == 'geraf':
                        # GOTCHA GeRaF (user decision 2026-09-22): the release configures the
                        # transmit amplitude for the dataset rather than learning it from the
                        # class default 0, which on GOTCHA's scale leaves the model predicting
                        # ~0 (validation RelMSE 1.0000, job 2156223). geraf_mf48_gotcha.json
                        # adds light_power_start; the collection lane keeps geraf_mf48.json.
                        args += ['--config', str(ROOT/'protocols/geraf_mf48_gotcha.json')]
                    elif method == 'sugavanam_ertin':
                        args += ['--config', str(ROOT/'protocols/se_g40_readout48.json')]
                    elif method == 'radarsplat':
                        args += ['--config', str(ROOT/'protocols/radarsplat_budget48.json')]
                    if stage == 'stage1':
                        args += ['--se-stage1-only']
                    if stage == 'stage2':
                        args += ['--resume', str(Path(previous['output_dir'])/'checkpoint_stage1_final.pt')]
                    dataset, planning = gotcha.make_plan(gotcha.parse_args(args))
                    entry = planning['plans'][0]
                    assert dataset.passes == list(range(1, 9)) or tuple(dataset.passes) == tuple(range(1, 9))
                    assert entry['training_pulse_selection']['pulses_per_sector'] == 16
                    assert 'frequency_stride2_all_roles_v2' in entry['output_dir']
                    script = 'train_gotcha_dataset_pvc.py'
                else:
                    args = ['--dataset-root', os.environ['RIFT_DATA_ROOT'],
                            '--output-root', str(root/'outputs/rift'), '--object', scene,
                            '--num-train', '2400', '--num-tx', '1', '--num-rx', '1',
                            '--method', method]
                    if method == 'geraf':
                        args += ['--geraf-source-config', str(ROOT/'protocols/geraf_mf48_1t1r.json')]
                    elif method == 'sugavanam_ertin':
                        args += ['--se-config', str(ROOT/'protocols/se_g40_readout48.json'), '--device', 'xpu']
                    elif method == 'radarsplat':
                        args += ['--radarsplat-recipe', 'budget48', '--device', 'xpu']
                    elif method == 'radar_fields':
                        args += ['--radar-fields-recipe', 'source-adapted-v3']
                    if stage == 'stage1':
                        args += ['--se-stage1-only']
                    if stage == 'stage2':
                        args += ['--resume', str(Path(previous['output_dir'])/'checkpoint_stage1_final.pt')]
                    parsed = collection.parse_args(args)
                    planning = collection.make_plan(parsed)
                    entry = planning['plans'][0]
                    assert planning['antenna_selection']['tx_indices'] == [0]
                    assert planning['antenna_selection']['rx_indices'] == [0]
                    # Publish only selected metadata before concurrent jobs start.
                    collection.base.write_selected_manifest(entry['role_manifest_path'],
                        role_manifest(entry['object'], parsed.num_train, planning['antenna_selection']))
                    script = 'train_rift_dataset_pvc.py'
                task = dict(key=key, scene=scene, method=method, stage=stage,
                            output_dir=entry['output_dir'],
                            command=[sys.executable, '-u', str(ROOT/script), *args],
                            depends_on=previous['key'] if stage == 'stage2' else None,
                            job_name='pvcprod-'+key.replace('sugavanam_ertin', 'se'), job_id=None)
                if scene == 'camry' and method == 'rift' and gotcha_rift_variant != 'default':
                    task['job_name'] = 'pvcprod-camry-rift-' + gotcha_rift_variant.replace('_', '-')
                if stage == 'stage2':
                    assert task['output_dir'] == previous['output_dir']
                save(root/'plans'/f'{key}.json', planning)
                tasks.append(task)
                previous = task
                print(f'Planned {key}', flush=True)
    assert len(tasks) == len(scenes)*(len(methods)+int('sugavanam_ertin' in methods))
    save(root/'campaign.json', dict(schema='rift_pvc_production_campaign_v1',
         created_at=datetime.now(timezone.utc).isoformat(), root=str(root),
         methods=list(methods), scenes=list(scenes),
         gotcha_rift_point_chunk=gotcha_rift_point_chunk,
         **({'gotcha_rift_variant': gotcha_rift_variant} if gotcha_rift_variant != 'default' else {}),
         pvc_backends=dict(radar_fields='upstream-tcnn-torchshim', tcnn_version='torchshim-2',
                           tcnn_precision='fp32', tcnn_weight_grad='bmm:256',
                           radarsplat='fork_torch_mirror_xpu_v1', ssim='fused_ssim_torch_v1',
                           strict_numerical_parity='not_certified'),
         slurm=dict(partition='pvc', gres='gpu:pvc:1', qos='normal', time='24:00:00',
                    cpus=8, memory='64G', account='158648339640', exclude='ac023'),
         acquisition=dict(rift_train=2400, tx_indices=[0], rx_indices=[0], gotcha_train=1500,
                          gotcha_polarizations=['hh'], gotcha_passes=list(range(1, 9)),
                          pulses_per_sector=16, frequency_stride=2, reserved_test='sealed'), tasks=tasks))


def submit(root):
    path = root/'campaign.json'
    campaign = json.loads(path.read_text())
    tasks = {task['key']: task for task in campaign['tasks']}
    # First stages are submitted before any dependents, so all 28 can start now.
    ordered = sorted(tasks.values(), key=lambda task: task['stage'] == 'stage2')
    for task in ordered:
        if task['job_id']:
            continue
        if task.get('submission_started'):
            raise RuntimeError(f"Unresolved submission for {task['key']}: inspect Slurm before retrying")
        command = ['sbatch', '--parsable', '--export=ALL', '--job-name', task['job_name'],
                   '--output', str(root/'logs'/f"{task['key']}-%j.slurm.log"),
                   '--chdir', str(root/'source')]
        if task['depends_on']:
            parent = tasks[task['depends_on']]['job_id']
            assert parent
            task['dependency'] = f'afterok:{parent}'
            command += ['--dependency', task['dependency'], '--kill-on-invalid-dep=yes']
        command += [str(root/'source/scripts_pvc/production_job.sbatch'), str(root), task['key']]
        task['submission_started'] = datetime.now(timezone.utc).isoformat()
        task['sbatch_command'] = command
        save(path, campaign)
        result = subprocess.run(command, text=True, capture_output=True, timeout=90)
        task['submission_stdout'], task['submission_stderr'] = result.stdout, result.stderr
        if result.returncode:
            save(path, campaign)
            raise RuntimeError(f"sbatch failed for {task['key']}: {result.stderr}")
        task['job_id'] = int(result.stdout.strip().split(';')[0])
        save(path, campaign)
        print(f"{task['job_id']} {task['key']} {task.get('dependency', 'ready')}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', '_plan', 'submit'])
    parser.add_argument('root', type=Path)
    parser.add_argument('--methods', nargs='+', choices=ALL_METHODS, default=METHODS)
    parser.add_argument('--scenes', nargs='+', choices=SCENES, default=SCENES)
    parser.add_argument('--gotcha-rift-variant', choices=sorted(GOTCHA_RIFT_VARIANTS), default='default',
                        help='Camry adaptive RIFT only: default recipe, or the opt-in RIFT-dataset NUFFT + full-native '
                             'loss control (--forward-evaluation nufft --loss-domain full_native; own root)')
    parser.add_argument('--gotcha-rift-point-chunk', type=int, default=16384,
                        help='--point-chunk for the Camry adaptive-RIFT command; pinned GOTCHA recipe 16384 (0 = trainer default 4096)')
    args = parser.parse_args()
    if len(args.methods) != len(set(args.methods)):
        parser.error('Select distinct methods')
    if len(args.scenes) != len(set(args.scenes)):
        parser.error('Select distinct scenes')
    if args.action == 'submit':
        submit(args.root.resolve())
    else:
        if args.gotcha_rift_variant != 'default' and ('camry' not in args.scenes or 'rift' not in args.methods):
            parser.error('--gotcha-rift-variant applies to the Camry adaptive-RIFT task; select --scenes camry --methods rift')
        {'prepare': prepare, '_plan': plan}[args.action](args.root.resolve(), args.methods, args.scenes,
                                                        args.gotcha_rift_point_chunk, args.gotcha_rift_variant)
