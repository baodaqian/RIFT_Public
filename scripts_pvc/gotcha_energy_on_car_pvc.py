#!/usr/bin/env python3
"""Where does an adaptive-RIFT GOTCHA scene put its energy relative to the Camry? (tuning campaign A58, CPU)

Tests the inference in docs/RIFT_GOTCHA_Tune.md section 5 (B56) that a scene fitted to per-pass
rotated data absorbs the rotations as spurious height and off-car structure. Here "energy" is each
active point's coefficient energy, the sum of |w|^2 over its unlocked SH bands. The positions and
energy are the ones the CIC geometry readout deposits (scripts/render_b787_vs_stl._point_sh_energy_field).
Distance is measured to the registered solid (``camry_xv20_solid_region_local.npz``: 2 cm occupancy,
region frame), using a Euclidean distance transform outside the solid, looked up at each point's nearest voxel.

Reported per checkpoint:
- energy shares inside the solid and in distance shells (m) around it;
- the off-car energy (distance > ``--tau``, 0.125 m, the geometry scorer's tau), split by where it lies:
  - ``above_roof``: a footprint column, above the column's top;
  - ``under_body``: a footprint column, between ground and the column's underside;
  - ``footprint_gap``: a footprint column, between underside and top but outside the solid;
  - ``below_ground``: more than tau below the lowest solid voxel (the tyre contact);
  - ``beside``: everything else;
- energy-weighted mean and median distance, and point counts.

``--field`` (A67) reads any method on the geometry scorer's 48^3 lattice instead, with the scorer's own
readouts and centres (``eval_gotcha_geometry_pvc.py``): ``rift`` the CIC energy field, ``spinr`` sigma^2,
``npz`` a precomputed field such as SE's ``se_energy`` or a data image. Each lattice cell counts as a point
at its centre. This is the readout that compares RIFT with the baselines; ``--run`` (point energies) is RIFT-only.

    python scripts_pvc/gotcha_energy_on_car_pvc.py --run M5b_ep16=PATH.pt --run CM5b_ep16=PATH.pt --output out.json
    python scripts_pvc/gotcha_energy_on_car_pvc.py --field F5full_ep24=rift:PATH.pt --field SpINR_B=spinr:PATH.pt \\
        --field SE_A=npz:PATH.npz:se_energy --output out.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

SHELLS = (0.0, 0.0625, 0.125, 0.25, 0.5, 1.0)
PLACES = ('above_roof', 'under_body', 'footprint_gap', 'below_ground', 'beside')


def point_energy(state, extent):
    """Active positions (region frame, m) and unlocked-band coefficient energy, as the CIC readout computes them."""
    sd = {k.split('.field.', 1)[1]: v for k, v in state.items() if '.field.' in k}
    active = sd['active_mask']
    order = sd['order'][active]
    positions = sd['anchors'][active] + sd['cell_half'][active] * torch.tanh(sd['delta_raw'][active])
    enabled = sd.get('support_bounds_enabled', torch.tensor(False))
    if bool(enabled):
        positions = torch.maximum(torch.minimum(positions, sd['support_max']), sd['support_min'])
    positions = positions.double().clamp(-extent, extent)
    unlocked = sd['basis_degree'][None, :] <= order[:, None]
    squared = sd['w_re'][active].double().square() + sd['w_im'][active].double().square()
    return positions.numpy(), torch.where(unlocked, squared, 0).sum(-1).numpy(), int(order.max())


def lattice_energy(spec, manifest):
    """Cell centres (region frame, m) and energy of one method on the scorer's lattice, as its F1 sweep reads it."""
    from scripts.render_b787_vs_stl import trilinear_sample_centers
    from scripts_pvc.eval_gotcha_geometry_pvc import GRID, npz_field, rift_field, spinr_field
    method, rest = spec.split(':', 1)
    extent = float(manifest['region']['half_extent_m'])
    if method == 'npz':
        path, key = rest.rsplit(':', 1)
        region, energy, info = npz_field(path, key)
        meta = dict(checkpoint=path, epoch=info.get('epoch'), cursor=None, readout_info=info)
    elif method in ('rift', 'spinr'):
        path = rest
        ck, fields = (rift_field if method == 'rift' else spinr_field)(path, extent)
        region = ck['dataset_contract']['region']
        if list(fields) != ['hh']:
            raise ValueError(f'{path}: expected one hh field, found {sorted(fields)}')
        energy, info = fields['hh']
        meta = dict(checkpoint=path, epoch=ck.get('epoch'), cursor=ck.get('cursor'), readout_info=info)
        del ck
    else:
        raise ValueError(f'unsupported field method {method!r} (rift, spinr, npz)')
    if json.loads(json.dumps(region)) != json.loads(json.dumps(manifest['region'])):
        raise ValueError(f'{spec}: region differs from the registered mesh region')
    axis = trilinear_sample_centers(extent, GRID, GRID)
    centres = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(-1, 3)
    energy = np.clip(np.asarray(energy, dtype=np.float64).reshape(-1), 0, None)
    meta.update(method=method, readout=f'lattice{GRID} (geometry scorer centres)', negative_clipped=True)
    return centres, energy, meta


def car_geometry(mesh_dir):
    solid = np.load(mesh_dir / 'camry_xv20_solid_region_local.npz')
    occupancy, axis, voxel = solid['occupancy'], solid['axis_m'], float(solid['voxel_m'])
    distance = ndimage.distance_transform_edt(~occupancy, sampling=voxel)
    footprint = occupancy.any(axis=2)
    z_index = np.arange(occupancy.shape[2])
    top = np.where(footprint, np.where(occupancy, z_index, -1).max(axis=2), -1)
    bottom = np.where(footprint, np.where(occupancy, z_index, occupancy.shape[2]).min(axis=2), -1)
    ground = axis[int(np.nonzero(occupancy.any(axis=(0, 1)))[0].min())]
    return dict(occupancy=occupancy, axis=axis, voxel=voxel, distance=distance, footprint=footprint,
                top=top, bottom=bottom, ground=float(ground))


def solid_distance(positions, car):
    """Lookup indices into the solid's grid and the distance (m) to the solid at each position's nearest voxel."""
    axis, voxel = car['axis'], car['voxel']
    idx = np.clip(np.rint((positions - axis[0]) / voxel).astype(int), 0, len(axis) - 1)
    ix, iy, iz = idx.T
    return idx, car['distance'][ix, iy, iz]


def rank_companions(centres, energy, car, tau):
    """Floor-free companions to the lattice near-car share (reviewer, after A67).

    A dense field (SpINR, a data image) carries a diffuse floor in every cell, which a sparse point field cannot, so
    the raw energy share mixes placement with sparsity. With k = the number of lattice cells within tau of the solid:
    ``uniform_near_share`` is what a flat field scores (k / cells); ``top_k_cells_near`` is the fraction of the k
    brightest cells that lie within tau (unweighted, no threshold); ``top_k_energy_near`` is their energy-weighted form.
    """
    _, d = solid_distance(centres, car)
    near = d <= tau
    k = int(near.sum())
    top = np.argsort(-energy, kind='stable')[:k]
    return dict(near_cells=k, cells=int(len(energy)), nonzero_cells=int((energy > 0).sum()),
                uniform_near_share=k / len(energy), top_k_cells_near=float(near[top].mean()),
                top_k_energy_near=float(energy[top][near[top]].sum() / max(energy[top].sum(), 1e-300)))


def partition(positions, energy, car, tau):
    idx, d = solid_distance(positions, car)
    ix, iy, iz = idx.T
    total = float(energy.sum())
    shells = {'inside': float(energy[d == 0].sum()) / total}
    edges = list(SHELLS[1:]) + [np.inf]
    lower = 0.0
    for upper in edges:
        name = f'{lower:g}-{upper:g}' if np.isfinite(upper) else f'>{lower:g}'
        shells[name] = float(energy[(d > lower) & (d <= upper)].sum()) / total if lower > 0 else \
            float(energy[(d > 0) & (d <= upper)].sum()) / total
        lower = upper
    off = d > tau
    in_print = car['footprint'][ix, iy]
    z = positions[:, 2]
    places = dict(
        below_ground=off & (z < car['ground'] - tau),
        above_roof=off & in_print & (iz > car['top'][ix, iy]),
        under_body=off & in_print & (iz < car['bottom'][ix, iy]) & (z >= car['ground'] - tau),
    )
    places['footprint_gap'] = off & in_print & ~places['above_roof'] & ~places['under_body'] & ~places['below_ground']
    places['beside'] = off & ~in_print & ~places['below_ground']
    order = np.argsort(d)
    cumulative = np.cumsum(energy[order]) / total
    return dict(
        total_energy=total, points=int(len(energy)),
        near_share=float(energy[~off].sum()) / total, off_share=float(energy[off].sum()) / total,
        shells=shells,
        off_by_place={k: float(energy[m].sum()) / total for k, m in places.items()},
        points_off_by_place={k: int(m.sum()) for k, m in places.items()},
        weighted_mean_distance_m=float((energy * d).sum()) / total,
        weighted_median_distance_m=float(d[order][np.searchsorted(cumulative, 0.5)]),
        energy_weighted_z_m=float((energy * z).sum()) / total,
        near_energy_weighted_z_m=float((energy[~off] * z[~off]).sum() / max(energy[~off].sum(), 1e-300)))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', action='append', default=[], help='LABEL=CHECKPOINT (RIFT point energies)')
    p.add_argument('--field', action='append', default=[],
                   help='LABEL=rift:CKPT | LABEL=spinr:CKPT | LABEL=npz:PATH:KEY (scorer lattice energy)')
    p.add_argument('--mesh-dir', type=Path, default=Path('data/meshes/camry_xv20_data_frame_box_v2'))
    p.add_argument('--tau', type=float, default=0.125)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    if not args.run and not args.field:
        p.error('give at least one --run or --field')
    manifest = json.loads((args.mesh_dir / 'manifest.json').read_text())
    extent = float(manifest['region']['half_extent_m'])
    car = car_geometry(args.mesh_dir)
    out = dict(schema='gotcha_energy_on_car_v1', mesh_dir=str(args.mesh_dir), tau_m=args.tau, extent_m=extent,
               ground_z_m=car['ground'], energy='active unlocked-band sum |w|^2 (the CIC readout energy)', runs={})
    for spec in args.run:
        label, path = spec.split('=', 1)
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        positions, energy, max_order = point_energy(checkpoint['model_state_dict'], extent)
        row = partition(positions, energy, car, args.tau)
        row.update(checkpoint=path, epoch=checkpoint.get('epoch'), cursor=checkpoint.get('cursor'), max_sh_order=max_order,
                   densify_recorded=bool((checkpoint.get('history') or [{}])[-1].get('optimizer', {}).get('densify')))
        out['runs'][label] = row
        places = ' '.join(f"{k} {row['off_by_place'][k]:.3f}" for k in PLACES)
        print(f"{label}: epoch {row['epoch']} points {row['points']} near {row['near_share']:.3f} off {row['off_share']:.3f} "
              f"| {places} | mean d {row['weighted_mean_distance_m']:.3f} m, median {row['weighted_median_distance_m']:.3f} m "
              f"| z {row['energy_weighted_z_m']:+.3f} m", flush=True)
        del checkpoint
    if args.field:
        out['fields'] = {}
    for spec in args.field:
        label, source = spec.split('=', 1)
        centres, energy, meta = lattice_energy(source, manifest)
        row = partition(centres, energy, car, args.tau)
        row.update(meta)
        row.update(rank_companions(centres, energy, car, args.tau))
        out['fields'][label] = row
        print(f"{label} [{meta['method']} lattice]: epoch {row['epoch']} near {row['near_share']:.3f} "
              f"(uniform {row['uniform_near_share']:.3f}; top-{row['near_cells']} cells near {row['top_k_cells_near']:.3f}, "
              f"energy {row['top_k_energy_near']:.3f}; nonzero {row['nonzero_cells']}) "
              f"off {row['off_share']:.3f} | mean d {row['weighted_mean_distance_m']:.3f} m, "
              f"median {row['weighted_median_distance_m']:.3f} m | z {row['energy_weighted_z_m']:+.3f} m", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()
