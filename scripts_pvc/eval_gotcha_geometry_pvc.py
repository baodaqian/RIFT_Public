#!/usr/bin/env python3
"""3D geometry of GOTCHA Camry reconstructions against the registered Camry XV20 stand-in (PVC).

Truth is ``scripts_pvc/prepare_gotcha_camry_mesh.py``'s output in the GOTCHA region frame
(the frame every GOTCHA model reconstructs in): the surface mesh and its solid occupancy.
It is a same-generation Camry model (user identification: XV20) registered to the dataset's
footprint workbook (RMS 3 cm), not a scan of the car; the ground and the rest of the parking
lot are not in it, so reconstructed ground/clutter energy counts as false positives.

The RIFT-dataset geometry protocol (``scripts/eval_b787_geometry_metrics.py``) is applied at
the Camry scale, reusing its functions:

* field methods (RIFT point-SH, SpINR): the method's energy on the evaluator's 48^3
  lattice over the region cube (RIFT: its own conservative CIC point-SH readout; SpINR: its
  signed field sigma queried at the lattice centres, energy sigma^2); magnitude sqrt(energy),
  min-max normalized, voxel centres above ``--fixed-threshold`` (0.2 as on the RIFT dataset);
  Chamfer against surface and volume truth, precision/recall/F1 at ``tau``, solid and
  shell IoU;
* surface methods (GeRaF): its SDF zero surface (``extract_zero_surface``), surface samples
  against mesh samples, Chamfer and precision/recall/F1 at ``tau`` (GeRaF's own protocol).

Scale: the RIFT dataset uses tau = one lattice pitch (6.25 mm = 0.3 m / 48) and an IoU cell
of 1/60 of the cube side; the same ratios here give tau = 10 m / 48 = 0.2083 m and a
0.1667 m IoU cell.

    python scripts_pvc/eval_gotcha_geometry_pvc.py --mesh-dir data/meshes/camry_xv20 --output OUT.json \\
        --run rift=ROOT_C7:camry-rift-full:CKPT --run spinr=ROOT_C3:camry-spinr-full:CKPT \\
        --run geraf=ROOT_G:camry-geraf-full:CKPT
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.eval_b787_geometry_metrics import (chamfer, predicted_points, prf, sample_surface_points,  # noqa: E402
                                               sample_volume_points, voxel_iou)
from scripts.render_b787_vs_stl import load_stl_vertices, trilinear_sample_centers  # noqa: E402

GRID = 48


def truth(mesh_dir, samples, seed):
    mesh_dir = Path(mesh_dir)
    manifest = json.loads((mesh_dir/'manifest.json').read_text())
    triangles = load_stl_vertices(mesh_dir/'camry_xv20_region_local.stl').reshape(-1, 3, 3)
    solid = np.load(mesh_dir/'camry_xv20_solid_region_local.npz')
    rng = np.random.default_rng(seed)
    surface = sample_surface_points(triangles, samples, rng)
    axis = solid['axis_m']
    volume = sample_volume_points(solid['occupancy'], axis, axis, axis, samples, rng)
    return manifest, surface, volume


def region_of(dataset_contract):
    return dataset_contract['region']


def rift_field(checkpoint, extent):
    from scripts.render_b787_vs_stl import _point_sh_energy_field
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    fields = {}
    for pol in sorted({k.split('.')[0] for k in ck['model_state_dict']}):
        prefix = f'{pol}.field.'
        state = {k[len(prefix):]: v for k, v in ck['model_state_dict'].items() if k.startswith(prefix)}
        volume, info = _point_sh_energy_field(state, extent, GRID)
        fields[pol] = (volume, info)
    return ck, fields


def spinr_field(checkpoint, extent):
    from rift.spinr_style import SpinrStyleINR, midpoint_grid
    from train import load_tensor_checkpoint
    from train_spinr_style_pvc import evaluate_neural_field_tiled
    ck = load_tensor_checkpoint(Path(checkpoint), map_location='cpu')
    points, _ = midpoint_grid(GRID, support_m=extent, device='cpu', dtype=torch.float64)
    fields = {}
    for pol in sorted({k.split('.')[0] for k in ck['model_state_dict']}):
        model = SpinrStyleINR(support_m=extent)
        model.load_state_dict({k[len(pol) + 1:]: v for k, v in ck['model_state_dict'].items() if k.startswith(pol + '.')})
        model.eval()
        with torch.no_grad():
            sigma = evaluate_neural_field_tiled(model, points, neural_point_tile=8192).double()
        fields[pol] = (sigma.square().reshape(GRID, GRID, GRID).numpy(),
                       dict(scene_repr='spinr_signed_field', spatial_readout='field_at_lattice_centres', energy='sigma^2'))
    return ck, fields


def geraf_surface(checkpoint, campaign_root, task, extent):
    from rift.geraf_source_data import GOTCHASourceData
    from rift.geraf_v1 import extract_zero_surface
    from rift_pvc.geraf_source_training import load_selected_models
    from scripts_pvc.eval_gotcha_heldout_pvc import load_run
    dataset, _, _ = load_run(campaign_root, task)
    data = GOTCHASourceData(dataset)
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    models, selected = load_selected_models(ck, data, torch.device('cpu'))
    surfaces = {}
    for pol, model in models.items():
        class MetricSDF(torch.nn.Module):
            def __init__(self, network):
                super().__init__()
                self.network = network
            def forward(self, points):
                return self.network(points / extent)[..., 0] * extent
        sdf = MetricSDF(model.sdf_network)
        try:
            vertices, faces, report = extract_zero_surface(sdf, extent, grid=GRID, chunk=16384)
        except ValueError as error:
            # No zero crossing in the cube is the method's result (no surface), scored as zero points.
            axis = torch.linspace(-extent, extent, GRID, dtype=next(sdf.parameters()).dtype)
            with torch.no_grad():
                field = torch.cat([sdf(block) for block in torch.cartesian_prod(axis, axis, axis).split(16384)])
            vertices, faces = np.empty((0, 3)), np.empty((0, 3), dtype=np.int64)
            report = dict(readout='native_sdf_zero_surface', failure=str(error), extent_m=extent, grid=GRID,
                          sdf_min_m=float(field.min()), sdf_max_m=float(field.max()))
        surfaces[pol] = (vertices, faces, report)
    return ck, surfaces, dataset.contract


def npz_field(path, key):
    """A precomputed energy field on the scorer's lattice (``gotcha_coherent_readout_pvc.py``), with its region."""
    data = np.load(path)
    meta = json.loads(str(data['meta']))
    energy = np.asarray(data[key], dtype=np.float64)
    if energy.shape != (GRID, GRID, GRID):
        raise ValueError(f'{path}:{key} is not a {GRID}^3 field')
    info = dict(readout=meta.get('readout'), key=key, units=meta.get('units'), epoch=meta.get('epoch'),
                checkpoint=meta.get('checkpoint'), freq_stride=meta.get('freq_stride'))
    return meta['region'], energy, info


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mesh-dir', type=Path, default=Path('data/meshes/camry_xv20'))
    parser.add_argument('--run', action='append', required=True, help='method=CAMPAIGN_ROOT:TASK:CHECKPOINT')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixed-threshold', type=float, default=0.2)
    parser.add_argument('--thresholds', type=float, nargs='*', default=[])
    parser.add_argument('--samples', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest, gt_surface, gt_volume = truth(args.mesh_dir, args.samples, args.seed)
    extent = float(manifest['region']['half_extent_m'])
    tau, iou_unit = 2 * extent / GRID, 2 * extent / 60
    centers_axis = trilinear_sample_centers(extent, GRID, GRID)
    centers = np.stack(np.meshgrid(centers_axis, centers_axis, centers_axis, indexing='ij'), -1).reshape(-1, 3)
    rows = []
    for item in args.run:
        method, spec = item.split('=', 1)
        root, task, checkpoint = spec.split(':', 2)
        if method == 'field':
            # --run field=NPZ:KEY:LABEL, e.g. a coherent backprojection readout (A52).
            region, energy, info = npz_field(root, task)
            if json.loads(json.dumps(region)) != json.loads(json.dumps(manifest['region'])):
                raise ValueError('field: readout region differs from the registered mesh region')
            fields = {'hh': (energy, info)}
            method = f'field:{task}:{checkpoint}' if checkpoint else f'field:{task}'
        elif method in ('rift', 'spinr'):
            ck, fields = (rift_field if method == 'rift' else spinr_field)(checkpoint, extent)
            if ck['dataset_contract']['region'] != manifest['region']:
                raise ValueError(f'{method}: checkpoint region differs from the registered mesh region')
        if method.startswith('field') or method in ('rift', 'spinr'):
            for pol, (energy, info) in fields.items():
                magnitude = np.sqrt(np.clip(energy, 0, None))
                normalized = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-30)
                for threshold in [args.fixed_threshold, *args.thresholds]:
                    prediction = predicted_points(normalized, centers, threshold)
                    row = dict(method=method, polarization=pol, checkpoint=checkpoint, threshold=threshold,
                               threshold_role='primary' if threshold == args.fixed_threshold else 'sweep',
                               protocol='thresholded_field', points=int(len(prediction)), readout=info)
                    if len(prediction):
                        row.update(surface=chamfer(prediction, gt_surface), volume=chamfer(prediction, gt_volume),
                                   prf=prf(prediction, gt_surface, tau),
                                   iou_solid=voxel_iou(prediction, gt_volume, iou_unit, -extent, extent),
                                   iou_shell=voxel_iou(prediction, gt_surface, iou_unit, -extent, extent))
                    rows.append(row)
        elif method == 'geraf':
            ck, surfaces, contract = geraf_surface(checkpoint, root, task, extent)
            if contract['region'] != manifest['region']:
                raise ValueError('geraf: dataset region differs from the registered mesh region')
            for pol, (vertices, faces, report) in surfaces.items():
                rng = np.random.default_rng(42)
                prediction = sample_surface_points(vertices[faces], args.samples, rng) if len(faces) else np.empty((0, 3))
                row = dict(method=method, polarization=pol, checkpoint=checkpoint, protocol='sdf_zero_surface',
                           points=int(len(prediction)), extraction=report)
                if len(prediction):
                    row.update(surface=chamfer(prediction, gt_surface), prf=prf(prediction, gt_surface, tau))
                rows.append(row)
        else:
            raise ValueError(f'unsupported method {method!r}')
    result = dict(schema='gotcha_camry_geometry_v1', truth=dict(mesh_dir=str(args.mesh_dir.resolve()),
                  source=manifest['source'], stand_in=manifest['stand_in'], registration=manifest['pose']),
                  lattice=dict(grid=GRID, extent_m=extent), tau_m=tau, iou_unit_m=iou_unit,
                  scale_rule='RIFT-dataset ratios: tau = one lattice pitch, IoU cell = cube side / 60',
                  samples=args.samples, rows=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True, default=lambda v: v.tolist() if hasattr(v, 'tolist') else str(v))
        handle.write('\n')
    for row in rows:
        f1 = row.get('prf', {}).get('f1')
        print(f"{row['method']:8s} {row['polarization']} {row['protocol']:17s} thr={row.get('threshold', '-')} "
              f"points={row['points']} f1={f1} l2_mm={row.get('surface', {}).get('l2_mm')}")


if __name__ == '__main__':
    main()
