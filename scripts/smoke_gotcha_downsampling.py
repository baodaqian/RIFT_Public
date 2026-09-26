#!/usr/bin/env python3
"""Bounded six-method GPU acquisition smoke, not production fitting or scoring.

One selected train/validation sector per pass, all 16 selected pulses in each,
stride 2 native bins, 64 spatial probe points. Reserved-test payloads stay sealed.
Each method runs in an isolated process with a timeout and incremental report.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
METHODS = ('rift', 'spinr', 'radar_fields', 'geraf', 'radarsplat', 'sugavanam_ertin')


def save(path, report):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def finite(value):
    import torch
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError('Nonfinite smoke tensor')


def probe_points(dataset, device):
    from rift.gotcha_training import _grid
    return _grid(4, dataset.region.half_extent_m, device).double()


def operator(method, dataset, view, role, device, contexts):
    import numpy as np
    import torch
    from rift.gotcha_training import native_forward, RangeReadout
    points = probe_points(dataset, device)
    weights = torch.linspace(.2, 1., len(points), device=device, dtype=torch.float64)
    observations = list(dataset.observations(*view, 'hh'))
    if method == 'geraf':
        acquisition = contexts['geraf'].acquisition(role, view, 'hh', dict(point_chunk=32, pair_chunk=4), device)
        response = contexts['geraf'].response(role, view, 'hh', device)
        return acquisition.matched_filter(response, points), len(observations)
    if method == 'radarsplat':
        from rift.radarsplat_gotcha import sector_power
        cache = contexts['radarsplat']
        index = next(i for i, key in cache.view_keys.items() if key == view)
        power = sector_power(dataset, *view, 'hh', cache.calibration[index], cache.config, device=device)
        return torch.from_numpy(power).to(device), len(observations)
    values = []
    for observation in observations:
        if method == 'rift':
            r = contexts['readout_'+device].for_observation(observation)
            prediction = native_forward(points, weights.to(torch.complex128), r['antenna'],
                r['frequencies'], observation.reference_range_m, point_chunk=32)
            # Projection coordinates may differ by a unitary SVD basis between
            # CPU/CUDA; compare the reconstructed native vector instead.
            prediction = RangeReadout.lift(RangeReadout.project(prediction, r), r)
        elif method == 'spinr':
            from rift.spinr_native import NativeKernel
            kernel = NativeKernel(observation, dataset.region, device=device, point_tile=32)
            prediction = kernel.render(points, weights, (2*dataset.region.half_extent_m/4)**3, 1.)
        elif method == 'radar_fields':
            from rift.radar_fields_gotcha import range_geometry, matched_range_power
            _, ranges = range_geometry(observation, dataset.region, 2, device)
            prediction = matched_range_power(observation, ranges)
        else:
            prediction = contexts['sugavanam_ertin'].render(points, weights.to(torch.complex128),
                observation, point_chunk=32, pair_chunk=1)
        finite(prediction)
        values.append(prediction.flatten())
    return torch.cat(values), len(observations)


def model_check(method, dataset, observation, device):
    """Small numerical model check; record coverage independently of operators."""
    import torch
    from rift.gotcha_training import RangeReadout
    points = probe_points(dataset, device)
    if method == 'radar_fields':
        from importlib.util import find_spec
        # Never substitute a Torch network for the released comparison model.
        if torch.cuda.get_device_capability() == (7, 0) and find_spec('tinycudann_bindings._70_C') is None:
            return dict(status='blocked_dependency', reason='Installed original TCNN lacks SM70/V100 binary; acquisition operator checked separately')
        from rift.radar_fields_upstream import original_module
        from rift.radar_fields_released import released_arguments
        model = original_module('radarfields.nn.models').RadarField(
            **released_arguments().model_settings, use_tcnn=True).to(device)
        for layer in model.modules():
            if hasattr(layer, 'jit_fusion'): layer.jit_fusion = False
        output = model(torch.rand(256, 3, device=device), torch.rand(256, 3, device=device))
        loss = output['alpha'].float().mean()+output['rd'].float().mean()
        coverage = 'released TCNN forward/backward on 256 synthetic query points'
    elif method == 'rift':
        import train_gotcha_dataset as cli
        from rift.gotcha_training import ChannelField, recipe_from_args
        recipe = recipe_from_args(cli.parse_args(['--granularity', '4', '--max-points', '64', '--sh-degree', '1']), 'rift')
        model = ChannelField('rift', dataset.region, recipe, device)
        readout = RangeReadout(dataset.region, device=device)
        r = readout.for_observation(observation)
        model.initialize_scale(observation, r)
        prediction, _ = model(observation, r)
        target = torch.as_tensor(observation.response, device=device)
        loss = (readout.project(prediction-target, r).abs().square().sum()
                / readout.project(target, r).abs().square().sum())
        coverage = 'adaptive G4/capacity64 degree0 field and native projected loss; one TRAIN pulse'
    elif method == 'spinr':
        from rift.spinr_style import SpinrStyleINR
        from rift.spinr_native import NativeKernel, bin_objective
        model = SpinrStyleINR(support_m=dataset.region.half_extent_m).to(device)
        field = model(points).reshape(-1)
        kernel = NativeKernel(observation, dataset.region, device=device)
        prediction = kernel.render(points, field, (2*dataset.region.half_extent_m/4)**3, 1.)
        loss = bin_objective(prediction, kernel.target_bins(observation.response),
                            float(abs(observation.response).dot(abs(observation.response))/len(observation.response)))
        coverage = 'full-width signed-real INR, G4 midpoint, exact selected native DFT loss; one TRAIN pulse'
    elif method == 'geraf':
        from rift.geraf_source import build_model, recipe_from_config
        model = build_model(recipe_from_config({}, dataset.region.half_extent_m), device)
        x = (points[:16]/dataset.region.half_extent_m).float().requires_grad_(True)
        sdf = model.sdf_network.sdf(x)
        grad, = torch.autograd.grad(sdf.sum(), x, create_graph=True)
        power = model.signal_network(x, x.new_empty((len(x), 0)), x.new_ones(len(x)))
        variance = model.deviation_network(x)
        loss = sdf.square().mean()+(grad.norm(dim=-1)-1).square().mean()+power.mean()+variance.mean()
        coverage = 'released full networks and second derivatives, bank2; source training loss/MF48 preparation not exercised'
    elif method == 'sugavanam_ertin':
        from rift.sugavanam_ertin_paper import PaperSDF, field_gradient
        model = PaperSDF(dataset.region.half_extent_m, initialization_std=.05).to(device)
        sdf, grad = field_gradient(model, points[:16].float(), create_graph=True)
        loss = sdf.square().mean()+(grad.norm(dim=-1)-1).square().mean()
        coverage = 'std0.05 full SDF and second derivatives; Stage1 operator checked separately, no constrained solve or Stage2 fit'
    else:
        from rift.radarsplat_release import load_cuda_reference, create_scene, ReleasedRenderer, release_loss
        from rift.radarsplat_b7873200 import RadarSplatGrid
        rendering, ssim = load_cuda_reference(device=device)
        grid = RadarSplatGrid(num_range_bins=16, range_resolution_m=.1, range_start_m=9.2,
            azimuth_start_deg=-7.2, azimuth_span_deg=14.4, output_azimuth_resolution_deg=.9,
            intermediate_azimuth_resolution_deg=.1, spectral_leakage_width_m=.7)
        model, _ = create_scene(scene_scale=1., scene_center=[10., 0., 0.], device=device, num_points=64)
        power, occ = ReleasedRenderer(rendering, 1.)(model, torch.eye(4, device=device), grid, 0,
                                                   torch.zeros(16, 16, device=device))
        loss = release_loss(power, occ, torch.full_like(power, .2), torch.full_like(occ, .3), model, ssim)['total']
        coverage = 'original CUDA rasterizer/filter/loss and fused SSIM, 64 synthetic Gaussians; production 112000 fit not exercised'
    finite(loss)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not any(bool(g.abs().max() > 0) for g in gradients):
        raise RuntimeError('No nonzero model gradients')
    for gradient in gradients: finite(gradient)
    torch.cuda.synchronize()
    return dict(status='passed', coverage=coverage, loss=float(loss.detach()),
                gradient_tensors=len(gradients), optimizer_steps=0)


def child(args):
    import numpy as np
    import torch
    from rift.gotcha_dataset import GOTCHADataset
    from rift.gotcha_training import RangeReadout
    report = dict(method=args.method, status='running', test_payload_read=False,
                  production_qualified=False, checks=[])
    save(args.output, report)
    started = time.monotonic()
    try:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
            raise RuntimeError('An allocated V100/SM70 is required')
        torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        report['gpu'] = dict(name=torch.cuda.get_device_name(), torch=str(torch.__version__),
                             memory_bytes=torch.cuda.get_device_properties(0).total_memory)
        dataset = GOTCHADataset(args.dataset_root, num_train=1500, pulses_per_sector=16, frequency_stride=2)
        report['dataset_identity'] = dataset.identity
        report['scope'] = 'first TRAIN and validation sector per pass, all selected pulses; G4/64-point operator probe'
        contexts = {'readout_cpu': RangeReadout(dataset.region), 'readout_cuda': RangeReadout(dataset.region, device='cuda')}
        if args.method == 'geraf':
            from rift.geraf_source_data import GOTCHASourceData
            contexts['geraf'] = GOTCHASourceData(dataset)
        elif args.method == 'sugavanam_ertin':
            from rift.sugavanam_ertin_acquisition import GOTCHAAcquisition
            contexts['sugavanam_ertin'] = GOTCHAAcquisition(dataset)
        elif args.method == 'radarsplat':
            from rift.radarsplat_gotcha import GOTCHAPowerCache
            contexts['radarsplat'] = GOTCHAPowerCache(dataset, 'hh', args.output.parent/'unused_metadata_cache',
                dict(azimuth_samples=11, elevation_samples=3, point_chunk=1024, frequency_chunk=128))
        first_observation = None
        for role in ('train', 'validation'):
            for p in dataset.passes:
                view = next(v for v in dataset.viewpoints(role) if v[0] == p)
                if first_observation is None: first_observation = next(dataset.observations(*view, 'hh'))
                prediction, count = operator(args.method, dataset, view, role, 'cuda', contexts)
                finite(prediction)
                item = dict(role=role, view=list(view), pulses=count,
                            frequencies=len(dataset.shards[p, 'hh'].frequencies_for_role(role)), elements=prediction.numel())
                if p == dataset.passes[0]:
                    reference, _ = operator(args.method, dataset, view, role, 'cpu', contexts)
                    residual = (prediction.detach().cpu().to(torch.complex128)-reference.detach().to(torch.complex128)).norm()
                    relative = float(residual/reference.norm().clamp_min(1e-30))
                    item['cpu_cuda_relative_l2'] = relative
                    if relative >= 1e-4: raise RuntimeError(f'CPU/CUDA operator mismatch: {relative}')
                report['checks'].append(item)
                save(args.output, report)
        for shard in dataset.shards.values():
            row = int(np.flatnonzero(shard.row_roles == 'test')[0])
            before = shard.response_reads
            try: shard.read(row)
            except PermissionError: pass
            else: raise RuntimeError('Test payload gate opened')
            assert shard.response_reads == before
        report['acquisition_status'] = 'passed'
        save(args.output, report)
        try:
            report['model_check'] = model_check(args.method, dataset, first_observation, 'cuda')
        except Exception:
            report['model_check'] = dict(status='failed', error=traceback.format_exc())
        torch.cuda.synchronize()
        report['peak_gpu_allocated_bytes'] = torch.cuda.max_memory_allocated()
        report['status'] = 'passed' if report['model_check']['status'] == 'passed' else 'passed_acquisition_model_incomplete'
    except Exception:
        report.update(status='failed', error=traceback.format_exc())
    report['wall_seconds'] = time.monotonic()-started
    save(args.output, report)
    return 0 if report['status'] == 'passed' else 2


def main(argv=None):
    from rift.gotcha_dataset import DEFAULT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--method', choices=METHODS)
    parser.add_argument('--seconds-per-method', type=int, default=85)
    args = parser.parse_args(argv)
    if not os.environ.get('SLURM_JOB_ID'): parser.error('Run inside the explicitly authorized GPU allocation')
    if not 10 <= args.seconds_per_method <= 85: parser.error('Per-method budget must be 10..85 seconds')
    if args.output.exists(): parser.error('Use a fresh output report')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.method: return child(args)
    report = dict(schema='gotcha_downsampling_gpu_smoke_v1', utc=datetime.now(timezone.utc).isoformat(),
        job_id=os.environ['SLURM_JOB_ID'], status='running', seconds_per_method=args.seconds_per_method,
        pulse_cap=16, frequency_stride=2, test_payload_read=False, production_qualified=False, methods={})
    save(args.output, report)
    for method in METHODS:
        output = args.output.parent/(method+'.json')
        command = [sys.executable, '-B', str(Path(__file__).resolve()), '--method', method,
                   '--dataset-root', str(args.dataset_root), '--output', str(output)]
        try:
            with (args.output.parent/(method+'.log')).open('w') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.seconds_per_method)
            item = json.loads(output.read_text()) if output.exists() else dict(status='failed_no_report')
            item['process_exit'] = result.returncode
        except subprocess.TimeoutExpired:
            item = json.loads(output.read_text()) if output.exists() else {}
            item['status'] = 'timeout'
        report['methods'][method] = item
        save(args.output, report)
        print(method, item['status'], flush=True)
    report['status'] = 'passed' if all(v['status'] == 'passed' for v in report['methods'].values()) else 'completed_with_limitations'
    save(args.output, report)
    return 0 if all(v.get('acquisition_status') == 'passed' for v in report['methods'].values()) else 2


if __name__ == '__main__':
    raise SystemExit(main())
