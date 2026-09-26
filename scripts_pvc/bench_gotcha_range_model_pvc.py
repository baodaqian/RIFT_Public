"""PVC card check of the GOTCHA amplitude law (``range_model``) on the batched lane.

Uses one Camry TRAIN pass-sector of the production acquisition (HH, 16-pulse
cap, frequency stride 2): its antennas, reference ranges and selected native
frequencies, with the production G48 point grid and point chunk. Reports
(1) on-device agreement of the batched sum2 kernel with autograd through the
per-pulse renderer, (2) sum2 == unit with weights pre-scaled by the RIFT
amplitude, and (3) forward+backward seconds for sum2 and unit. Exits 1
(SMOKE_GATE=FAIL) on an XPU-to-CPU fallback warning or when (1)/(2) exceed
their tolerances; timing is reported, not gated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import warnings

import torch

from rift.gotcha_dataset import GOTCHADataset, load_region
from rift.gotcha_training import range_amplitude
from rift_pvc import accelerator
from rift_pvc import gotcha_training as pvc
from rift_pvc.gotcha_batched import batched_native_forward


def main():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        report = run()
    for w in caught:
        print(f'warning: {w.category.__name__}: {w.message}', file=sys.stderr, flush=True)
    report['fallback_warnings'] = [str(w.message) for w in caught if 'fallback' in str(w.message).lower()]
    check = report['device_check']
    failures = [name for name, bad in (
        ('xpu_to_cpu_fallback', bool(report['fallback_warnings'])),
        ('forward_rel', not check['forward_rel'] <= 1e-10),
        ('weight_grad_rel', not check['weight_grad_rel'] <= 1e-10),
        ('position_grad_rel', not check['position_grad_rel'] <= 1e-5),   # float32 parameters
        ('sum2_equals_scaled_unit_rel', not report['sum2_equals_scaled_unit_rel'] <= 1e-12)) if bad]
    report['gate'] = 'FAIL: ' + ', '.join(failures) if failures else 'PASS'
    print(json.dumps(report, indent=2), flush=True)
    print(f"SMOKE_GATE={report['gate']}", flush=True)
    return 1 if failures else 0


def run():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--granularity', type=int, default=48)
    p.add_argument('--point-chunk', type=int, default=16384)
    p.add_argument('--repeats', type=int, default=5)
    args = p.parse_args()
    device = accelerator.device()
    ds = GOTCHADataset(args.dataset_root, passes=range(1, 9), polarizations=('hh',), region=load_region('camry', None),
                       num_train=1500, pulses_per_sector=16, frequency_stride=2)
    p_id, sector = ds.viewpoints('train')[0]
    obs = list(ds.observations(p_id, sector, 'hh'))
    readout = pvc.RangeReadout(ds.region, device=device)
    rs = [readout.for_observation(o) for o in obs]
    antennas = torch.stack([r['antenna'] for r in rs])
    refs = torch.tensor([o.reference_range_m for o in obs], dtype=torch.float64, device=device)
    freqs = rs[0]['frequencies']
    points = pvc._grid(args.granularity, ds.region.half_extent_m, device).float()
    g = torch.Generator(device='cpu').manual_seed(0)
    weights = torch.complex(torch.randn(len(obs), len(points), generator=g),
                            torch.randn(len(obs), len(points), generator=g)).to(device) * 1e-3
    report = dict(device=accelerator.describe(), pass_id=p_id, sector=int(sector), pulses=len(obs),
                  frequencies=int(freqs.numel()), points=len(points), point_chunk=args.point_chunk)

    # (1) device correctness on a subset: batched custom backward vs per-pulse autograd.
    x, w = points[:4096].clone(), weights[:4, :4096].clone()
    xl, wl = x.clone().requires_grad_(), w.clone().requires_grad_()
    outs = [pvc.native_forward(xl, wl[i], antennas[i], freqs, float(refs[i]), point_chunk=1024, range_model='sum2')
            for i in range(4)]
    target = torch.stack(outs).detach() * 0.5
    ((torch.stack(outs) - target).abs().square().sum()).backward()
    xb, wb = x.clone().requires_grad_(), w.clone().requires_grad_()
    out = batched_native_forward(xb, wb, antennas[:4], refs[:4], freqs, point_chunk=1024, range_model='sum2')
    ((out - target).abs().square().sum()).backward()
    rel = lambda a, b: float((a - b).abs().max() / b.abs().max())
    report['device_check'] = dict(forward_rel=rel(out.detach(), torch.stack(outs).detach()),
                                  weight_grad_rel=rel(wb.grad, wl.grad), position_grad_rel=rel(xb.grad, xl.grad))

    # (2) sum2 is unit with weights scaled by the RIFT-dataset amplitude.
    with torch.no_grad():
        dist = torch.linalg.vector_norm(points.double()[None] - antennas[:, None], dim=-1)
        s2 = batched_native_forward(points, weights, antennas, refs, freqs, point_chunk=args.point_chunk, range_model='sum2')
        un = batched_native_forward(points, weights * range_amplitude(dist, 'sum2'), antennas, refs, freqs,
                                    point_chunk=args.point_chunk)
    report['sum2_equals_scaled_unit_rel'] = rel(s2, un)

    # (3) timing of one sector's forward + backward (with per-pulse distance gradients).
    target = s2.detach() * 0.5
    timing = {}
    for model in ('unit', 'sum2', 'unit', 'sum2'):
        ww = weights.clone().requires_grad_()
        xx = points.clone().requires_grad_()
        seconds = []
        for _ in range(args.repeats + 1):
            grad_d = torch.zeros(len(obs), len(points), dtype=torch.float64, device=device)
            accelerator.synchronize()
            start = time.perf_counter()
            out = batched_native_forward(xx, ww, antennas, refs, freqs, point_chunk=args.point_chunk,
                                         pulse_grad_d=grad_d, range_model=model)
            (out - target).abs().square().sum().backward()
            accelerator.synchronize()
            seconds.append(time.perf_counter() - start)
        timing.setdefault(model, []).append(sum(seconds[1:]) / args.repeats)
    report['forward_backward_seconds'] = {k: [round(v, 4) for v in vs] for k, vs in timing.items()}
    report['sum2_over_unit'] = round(min(timing['sum2']) / min(timing['unit']), 4)
    return report


if __name__ == '__main__':
    raise SystemExit(main())
