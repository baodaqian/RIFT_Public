#!/usr/bin/env python3
"""Data for the RIFT problem-illustration schematic (B787; CPU). Reads no reserved-test or unused response.

1. A viewpoint path: every dataset view within ``--band`` deg of the elevation ring ``--elevation`` about the
   aircraft's up axis (+y in the B787 frame; nose +z). RIFT renders all of them from geometry alone (the
   evaluator's own renderer, ``export_rift_dataset_sphere_response_pvc.RiftAdapter``, at the paper checkpoint).
   Measured responses are taken only for TRAIN and VALIDATION views, from the sphere-response readout
   (``<root>/b787/measured.npz``); held-out geometry is public metadata, its responses stay sealed.
   Saved per ring view: the full 600-frequency complex response (RIFT everywhere; measured where it exists), its
   complex value at the band centre S_v(f_c) (the figure's signal), and the received power 10 log10 sum_f |S|^2.
2. v*: the VALIDATION view on the ring whose RIFT complex RelMSE is closest to the median over all VALIDATION
   views (a typical view, not a best case).
3. The learned field for the centre box: active point positions (plan view) with their energies and unlocked SH
   degrees L_p, and for the strongest points (``--candidates`` with L_p >= 1 and a quarter as many isotropic ones)
   the magnitude of their learned direction-dependent response |rho_p(u)| over directions u in the horizontal
   plane; the plot picks which ones to draw.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

CHECKPOINT = ('/scratch/group/p.cis261724.000/RIFT_pvc_runs/production_20260921_35jobs/outputs/rift/train2400/'
              '1t1r_2010b2bbe725/b787/rift/checkpoint_best.pth.tar')
MANIFEST = ('/scratch/group/p.cis261724.000/RIFT_pvc_runs/production_20260921_35jobs/outputs/rift/train2400/'
            '1t1r_2010b2bbe725/b787/role_manifest.json')
DATASET = Path('/scratch/user/u.db364833/RIFT_runs/h100_smoke_20260920_6d58645/inputs/RIFT_dataset')
UP, NOSE = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])     # B787 frame of the paper's view figures
LEFT = np.cross(UP, NOSE)                                             # +x


def power_db(spectrum):
    return 10 * np.log10(np.sum(np.abs(np.asarray(spectrum, dtype=np.complex128)) ** 2, axis=-1))


@torch.no_grad()
def main(argv=None):
    from rift.radar_fields_dataset import from_collection_arrays
    from rift.rift_dataset import evaluation_role_indices, load_object_contract, resolve_object_inputs
    from scripts.eval_b787_range_power import theta_phi
    from scripts_pvc.export_rift_dataset_sphere_response_pvc import RiftAdapter
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, default=Path('/scratch/user/u.db364833/RIFT_runs/sphere_response_20260924'))
    p.add_argument('--elevation', type=float, default=35.0)
    p.add_argument('--band', type=float, default=1.5)
    p.add_argument('--candidates', type=int, default=400)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    device = torch.device('cpu')

    npz_path, manifest = map(str, resolve_object_inputs(object_name='b787', dataset_root=DATASET, npz_path=None,
                                                        role_manifest_path=MANIFEST))
    public, contract = load_object_contract(npz_path, manifest, response_roles=('train', 'validation'))
    arrays = from_collection_arrays(public, contract)
    roles = {role: set(evaluation_role_indices(contract, role).tolist()) for role in ('train', 'validation')}
    positions = np.asarray(arrays.viewpoint_positions, dtype=np.float64)
    unit = positions / np.linalg.norm(positions, axis=1, keepdims=True)
    elevation = np.degrees(np.arcsin(np.clip(unit @ UP, -1, 1)))
    azimuth = np.degrees(np.arctan2(unit @ LEFT, unit @ NOSE)) % 360.0
    band = np.flatnonzero(np.abs(elevation - args.elevation) <= args.band)
    band = band[np.argsort(azimuth[band])]
    role_of = np.array(['train' if v in roles['train'] else 'validation' if v in roles['validation'] else 'heldout'
                        for v in band.tolist()])

    with np.load(args.root / 'b787' / 'measured.npz') as saved:
        measured = dict(zip(saved['view_indices'].tolist(), saved['spectrum']))
    with np.load(args.root / 'b787' / 'rift.npz') as saved:
        rift_rel = dict(zip(saved['view_indices'].tolist(), saved['coherent_rel_mse'].tolist()))
        val_median = float(np.median(saved['coherent_rel_mse'][saved['roles'] == 'validation']))
    candidates = [v for v, r in zip(band.tolist(), role_of) if r == 'validation']
    v_star = min(candidates, key=lambda v: abs(rift_rel[v] - val_median))

    adapter = RiftAdapter(CHECKPOINT, contract, arrays, device, argparse.Namespace())
    predicted = []
    for view in band.tolist():
        rx = torch.as_tensor(arrays.rx_pos[view], dtype=torch.float32)
        tx = torch.as_tensor(arrays.tx_pos[view], dtype=torch.float32)
        prediction = adapter.predict(view, 0, rx, tx, positions[view])
        predicted.append(prediction.permute(2, 1, 0).reshape(-1, arrays.num_freq)[0].cpu().numpy())
    predicted = np.stack(predicted)
    measured_db = np.array([power_db(measured[v]) if r != 'heldout' else np.nan for v, r in zip(band.tolist(), role_of)])
    check = [abs(power_db(predicted[i]) - power_db(measured[v])) for i, v in enumerate(band.tolist()) if role_of[i] != 'heldout']
    print(f'ring el {args.elevation} +- {args.band} deg: {len(band)} views ({(role_of == "train").sum()} train, '
          f'{(role_of == "validation").sum()} validation, {(role_of == "heldout").sum()} held out); v* = {v_star} '
          f'(RIFT complex RelMSE {rift_rel[v_star]:.4%}, validation median {val_median:.4%}); '
          f'max |RIFT - measured| power on scored views {max(check):.3f} dB', flush=True)

    # learned field: plan-view positions (image right = +z nose, up = +x), energies, lobes of a few points
    model = adapter.model
    active = model.active_mask
    pos = model.positions()[active].cpu().numpy()
    order = model.order[active].cpu().numpy()
    energy = (model.w_re[active] ** 2 + model.w_im[active] ** 2).sum(1).cpu().numpy()
    angles = np.radians(np.arange(0, 360, 3.0))
    directions = np.stack([np.sin(angles), np.zeros_like(angles), np.cos(angles)], 1)   # horizontal plane
    lobes = []
    for u in directions:
        theta, phi = theta_phi(u * 10.0)
        _, weights = model.active_scatterers(torch.tensor([[theta]], dtype=torch.float32),
                                             torch.tensor([[phi]], dtype=torch.float32))
        lobes.append(weights.abs().cpu().numpy())
    lobes = np.stack(lobes, 1)                                        # [points, directions]
    z, x = pos[:, 2], pos[:, 0]
    # lobe candidates: the strongest points, most of them with unlocked degree >= 1 (the plot picks among them)
    strong_sh = np.flatnonzero(order >= 1)
    strong_sh = strong_sh[np.argsort(energy[strong_sh])[::-1][:args.candidates]]
    strong_iso = np.flatnonzero(order == 0)
    strong_iso = strong_iso[np.argsort(energy[strong_iso])[::-1][:args.candidates // 4]]
    candidates = np.concatenate([strong_sh, strong_iso])
    # their learned SH coefficients (degree-major real basis, rift.spherical_harmonics.real_sh_basis), masked to each
    # point's unlocked degree L_p, so a plot can evaluate rho_p(u) over the whole sphere of directions
    n_basis = (int(model.max_degree) + 1) ** 2
    basis_degree = model.basis_degree[:n_basis].cpu().numpy()
    coefficients = (model.w_re[active][candidates, :n_basis] + 1j * model.w_im[active][candidates, :n_basis]).cpu().numpy()
    coefficients = coefficients * (basis_degree[None, :] <= order[candidates][:, None])
    print('orders:', {int(k): int(v) for k, v in zip(*np.unique(order, return_counts=True))},
          f'| lobe candidates: {len(strong_sh)} with L_p >= 1, {len(strong_iso)} isotropic', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    freqs = np.asarray(__import__('rift.radar_fields_dataset', fromlist=['build_frequency_grid'])
                       .build_frequency_grid(arrays.metadata))
    centre = int(np.argmin(np.abs(freqs - float(arrays.metadata['radar_fc_hz']))))
    measured_spectra = np.stack([measured[v] if r != 'heldout' else np.full(arrays.num_freq, np.nan + 0j)
                                 for v, r in zip(band.tolist(), role_of)]).astype(np.complex64)
    np.savez(args.output, ring_views=band, ring_roles=role_of, ring_azimuth_deg=azimuth[band],
             ring_elevation_deg=elevation[band], ring_unit=unit[band], rift_power_db=power_db(predicted), rift_spectra=predicted.astype(np.complex64),
             measured_spectra=measured_spectra, centre_index=centre, centre_hz=freqs[centre],
             measured_power_db=measured_db, v_star=v_star, point_plan=np.stack([z, x], 1), point_xyz=pos.astype(np.float32), point_energy=energy,
             scene_extent_m=float(adapter.o.resolve_scene_extent(torch.load(CHECKPOINT, map_location='cpu',
                                                                          weights_only=False))),
             point_order=order, lobe_index=candidates, lobe_angles_rad=angles, lobe_magnitude=lobes[candidates],
             lobe_coefficients=coefficients, lobe_basis_degree=basis_degree,
             meta=json.dumps(dict(checkpoint=CHECKPOINT, elevation_deg=args.elevation, band_deg=args.band,
                                  v_star=v_star, v_star_rift_complex_rel_mse=rift_rel[v_star],
                                  validation_median=val_median, frame='B787: nose +z, up +y, left +x',
                                  plan_view='image right = +z (nose), up = +x')))
    print('wrote', args.output, flush=True)


if __name__ == '__main__':
    main()
