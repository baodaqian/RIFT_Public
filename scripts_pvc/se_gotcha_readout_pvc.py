#!/usr/bin/env python3
"""Geometry readouts of a Sugavanam-Ertin run on GOTCHA Camry (camry_box_v2 setup; BASELINES_CAMRY_BOX.md).

``field``   Stage 1: the sub-aperture fields' incoherent energy sum_g |x_g(q)|^2 at the G40 grid points
            (region frame), deposited by the geometry scorer's conservative CIC on its 48^3 lattice and
            written as an npz for ``eval_gotcha_geometry_pvc.py --run field=NPZ:se_energy:LABEL``
            (the same readout family as RIFT's CIC point energy). The region is the checkpoint's own.
``fit``     Stage 1: RIFT's role-fit numbers (full-native RelMSE, energy ratio e, correlation rho, e/rho^2)
            of the sub-aperture fields on the TRAIN views (each rendered by its own sub-aperture's field, the
            data the fields were fitted to) and on the VAL views (each by the field of the sub-aperture its
            direction falls in, the workflow's own validation assignment), pooled over pulses and bins in the
            TRAIN-rms units the fields live in. The dataset is rebuilt by the frontend from the run's
            arguments (after ``--gotcha``) and bound to the checkpoint's acquisition identity.
``surface`` Stage 2: the exported zero level set (``surface.npz``, region frame) against the registered
            mesh (``--mesh-dir``, region-local STL): area-weighted samples on both (one generator, seed 42,
            prediction first) and the evaluator's chamfer and prf at ``--tau`` (the GOTCHA scorer's 0.125 m).
``field`` and ``surface`` read no radar response; ``fit`` reads TRAIN and VAL rows only (TEST stays sealed).

    python scripts_pvc/se_gotcha_readout_pvc.py field SE_RUN/checkpoint_latest.pt OUT.npz
    python scripts_pvc/se_gotcha_readout_pvc.py surface SE_RUN/surface.npz OUT.json
    python scripts_pvc/se_gotcha_readout_pvc.py fit SE_RUN/checkpoint_latest.pt OUT.json --gotcha <frontend args>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rift.sugavanam_ertin_paper_workflow import grid_points  # noqa: E402
from scripts.eval_b787_geometry_metrics import chamfer, prf, sample_surface_points  # noqa: E402
from scripts.eval_scene_geometry import deposit_points  # noqa: E402
from scripts.render_b787_vs_stl import load_stl_vertices  # noqa: E402

GRID = 48


def field(checkpoint_path, output):
    ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    region = ck['acquisition']['contract']['region']
    extent = float(region['half_extent_m'])
    points = grid_points(extent, ck['recipe']['granularity'])
    fields = ck['fields'].to(torch.complex128)
    energy = fields.abs().square().sum(0)
    volume = deposit_points(points, energy, extent, GRID).numpy()
    axis = (np.arange(GRID) + 0.5) * (2 * extent / GRID) - extent
    meta = dict(schema='se_gotcha_stage1_energy_v1', readout='sum over sub-apertures of |x_g|^2 at the G'
                f"{ck['recipe']['granularity']} grid points, CIC-deposited on the scorer's {GRID}^3 lattice",
                checkpoint=str(checkpoint_path), phase=ck.get('phase'), subapertures=int(fields.shape[0]),
                granularity=int(ck['recipe']['granularity']), region=region, extent_m=extent, units='relative')
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, axis_m=axis, se_energy=volume, meta=json.dumps(meta))
    print(json.dumps(dict(output=str(output), energy_total=float(energy.sum()), top1000_share=float(
        np.sort(volume.ravel())[::-1][:1000].sum() / volume.sum()))))


def surface(surface_path, output, mesh_dir, samples, tau):
    s = np.load(surface_path, allow_pickle=True)
    rng = np.random.default_rng(42)
    tris = s['vertices'][s['faces']]
    pred = sample_surface_points(tris, samples, rng) if len(tris) else np.zeros((0, 3))
    truth = sample_surface_points(load_stl_vertices(Path(mesh_dir) / 'camry_xv20_region_local.stl').reshape(-1, 3, 3),
                                  samples, rng)
    report = dict(schema='se_gotcha_surface_geometry_v1', surface=str(surface_path), mesh_dir=str(mesh_dir),
                  tau_m=tau, samples=samples, vertices=int(len(s['vertices'])), faces=int(len(s['faces'])),
                  prediction_bounds_m=[pred.min(0).tolist(), pred.max(0).tolist()] if len(pred) else None,
                  prf=prf(pred, truth, tau), surface_metrics=chamfer(pred, truth))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(report))


def role_numbers(sums):
    e = sums['prediction'] / sums['target']
    rho = sums['cross'] / max(np.sqrt(sums['prediction'] * sums['target']), 1e-300)
    return dict(full_native_rel_mse=sums['error'] / sums['target'], energy_ratio=e, correlation=rho,
                e_over_rho2=(e / rho ** 2 if rho else None), views=sums['views'], samples=sums['samples'])


def fit(checkpoint_path, output, gotcha_args):
    import train_gotcha_dataset_pvc as frontend
    from rift.sugavanam_ertin_acquisition import GOTCHAAcquisition, digest
    from rift_pvc.sugavanam_ertin_batched import _collect, fourier_forward_budget
    from rift_pvc.sugavanam_ertin_spgl1 import pinned_partition
    ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    dataset, plan = frontend.make_plan(frontend.parse_args(gotcha_args))
    acquisition = GOTCHAAcquisition(dataset)
    if digest(acquisition.identity) != digest(ck['acquisition']):
        raise ValueError('The rebuilt acquisition differs from the checkpoint\'s')
    recipe = ck['recipe']
    partition, validation_assignments, _ = pinned_partition(acquisition, recipe, Path(checkpoint_path).parent)
    points = grid_points(acquisition.extent, recipe['granularity'])
    fields, rms = ck['fields'], ck['statistics']['rms']
    report = {}
    for role, assignments in (('train', partition.assignments), ('validation', validation_assignments)):
        sums = dict(error=0., target=0., prediction=0., cross=0., views=0, samples=0)
        with torch.no_grad():
            for i, key in enumerate(acquisition.keys[role]):
                weights = fields[int(assignments[i])]
                for g in _collect(acquisition, acquisition.observations(key, role=role), 'cpu'):
                    pred = fourier_forward_budget(points, weights, g['f'], g['d'], g['r'], g['a'], cc=g['cc'],
                                                  point_chunk=recipe['point_chunk'], pair_chunk=recipe['pair_chunk'])
                    target = g['t'] / rms
                    sums['error'] += float((pred - target).abs().square().sum())
                    sums['target'] += float(target.abs().square().sum())
                    sums['prediction'] += float(pred.abs().square().sum())
                    sums['cross'] += float((pred.conj() * target).real.sum())
                    sums['samples'] += target.numel()
                sums['views'] += 1
        report[role] = role_numbers(sums)
        print(role, json.dumps(report[role]), flush=True)
    report.update(schema='se_gotcha_stage1_fit_v1', checkpoint=str(checkpoint_path), phase=ck.get('phase'),
                  note='TRAIN: each view by its own sub-aperture field (fitted to the 1% Eq. 4 budget); VAL: by the '
                       'field of the sub-aperture its direction falls in; stage1_subaperture_diagnostic_not_SDF_NVS')
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=1) + '\n')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=['field', 'surface', 'fit'])
    p.add_argument('source', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--mesh-dir', type=Path, default=ROOT / 'data/meshes/camry_xv20_data_frame_box_v2')
    p.add_argument('--samples', type=int, default=20000)
    p.add_argument('--tau', type=float, default=0.125)
    p.add_argument('--gotcha', nargs=argparse.REMAINDER, help='fit: the run\'s frontend arguments (must be last)')
    args = p.parse_args(argv)
    if args.command == 'fit':
        fit(args.source, args.output, args.gotcha)
    elif args.command == 'field':
        field(args.source, args.output)
    else:
        surface(args.source, args.output, args.mesh_dir, args.samples, args.tau)


if __name__ == '__main__':
    main()
