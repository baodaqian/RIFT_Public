#!/usr/bin/env python3
"""Does a GOTCHA adaptive-RIFT checkpoint fit its data? Fit, recency and Adam-step diagnostics (CPU).

For sampled TRAIN and VALIDATION pass-sectors it renders the checkpoint through the trainer's own
``ChannelField`` and ROI projection and reports, pooled over pulses:
- ``projected_rel_mse``, the metric the trainer selects on;
- ``energy_ratio``, predicted / measured projected energy;
- ``correlation``, Re<yhat, y> / (|yhat| |y|).
With energy ratio e and correlation rho, RelMSE = 1 + e - 2 rho sqrt(e): a model with the right
energy and no phase agreement scores about 2, and predicting zero scores 1.

Full-native energy ratio and correlation are reported too (the loss domain of ``--loss-domain
full_native`` recipes, including the box-isolated filtered target, which also needs ``--shard-root``
and ``--region-config``).

``--recency`` also scores the last and first sectors of the fixed B787-schedule training order.
Low correlation on sectors trained an epoch ago, next to higher correlation on the last ones, means
each update overwrites the scene. The Adam state gives the median sqrt(v_hat) / eps per parameter
group: well below 1 is B787's eps-damped regime (steps proportional to the gradient), well above 1
full Adam-normalized steps.

``--curvature N`` estimates, on N TRAIN sectors, the largest eigenvalue of the Hessian of one
update's data loss in the SH-coefficient block (power iteration). The prediction is linear in the
coefficients at fixed positions and gain, so the Hessian is exactly 2 Re(J^H J) / (n P) and does not
depend on the data or the coefficient values; it scales as |gain|^2. In the eps-damped regime Adam's
step is (lr / eps) m, so (lr / eps) lambda_max is the step's stability number: above 2 one plain step
on the sector overshoots along its top direction; about 2 (1 + beta1) / (1 - beta1) = 38 bounds heavy
EMA momentum on a fixed quadratic. It is reported at the checkpoint's gain and rescaled to the start
gain, at the checkpoint's SH orders and with every order set to 0 (orthonormal SH: an anchor's bands
up to L act through sum_k Y_k w_k within a sector, so the ratio is about (L + 1)^2).

Reads TRAIN/VALIDATION responses only; test stays sealed (the dataset adapter refuses it).

    source .local-setup/activate-pvc.sh
    python scripts_pvc/gotcha_rift_fit_check.py <checkpoint_best.pt> --sectors 24 --recency 8 --output fit.json
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rift.gotcha_dataset import GOTCHADataset, load_region  # noqa: E402
from rift.gotcha_training import RIFT_DATASET_REFERENCE, with_legacy_keys  # noqa: E402
from rift_pvc.gotcha_training import ChannelField, RangeReadout, _target  # noqa: E402
from rift_pvc import gotcha_nufft as nufft  # noqa: E402


def dataset_for(checkpoint, dataset_root, shard_root=None, region_config=None):
    contract = checkpoint['dataset_contract']
    selection = contract.get('training_pulse_selection') or {}
    cap = int(selection.get('pulses_per_sector', 0) or 0) if isinstance(selection, dict) else 0
    stride = int((contract.get('frequency_selection') or {}).get('stride', 1) or 1)
    split = contract['split']
    num_train = split.get('training_selection', {}).get('num_train') if 'training_selection' in split else None
    ds = GOTCHADataset(dataset_root, shard_root=shard_root, passes=contract['passes'],
                       polarizations=contract['polarizations'],
                       region=load_region(contract['region']['name'], region_config), num_train=num_train,
                       pulses_per_sector=cap, frequency_stride=stride)
    if ds.identity != checkpoint['dataset_identity']:
        raise ValueError('Rebuilt dataset identity differs from the checkpoint; pass the matching dataset root')
    return ds


def score(head, readout, ds, views, polarization):
    rows = []
    with torch.no_grad():
        for p, s in views:
            err = energy = predicted = cross = full_err = full_energy = full_predicted = full_cross = 0.
            for obs in ds.observations(p, s, polarization):
                r = readout.for_observation(obs)
                native, rendered = _target(obs, 'cpu'), head(obs, r)[0]
                full_err += float((rendered - native).abs().square().sum())
                full_energy += float(native.abs().square().sum())
                full_predicted += float(rendered.abs().square().sum())
                full_cross += float((rendered.conj() * native).sum().real)
                y = RangeReadout.project(native, r)
                yhat = RangeReadout.project(rendered, r)
                err += float((yhat - y).abs().square().sum())
                energy += float(y.abs().square().sum())
                predicted += float(yhat.abs().square().sum())
                cross += float((yhat.conj() * y).sum().real)
            rows.append(dict(view=[p, s], err=err, energy=energy, predicted=predicted, cross=cross,
                             full_err=full_err, full_energy=full_energy, full_predicted=full_predicted,
                             full_cross=full_cross))
    return rows


def pooled(rows):
    err, energy, predicted, cross = (sum(r[k] for r in rows) for k in ('err', 'energy', 'predicted', 'cross'))
    full_energy, full_predicted, full_cross = (sum(r[k] for r in rows) for k in ('full_energy', 'full_predicted', 'full_cross'))
    full = sum(r['full_err'] for r in rows) / full_energy
    return dict(sectors=len(rows), projected_rel_mse=err / energy, full_native_rel_mse=full,
                energy_ratio=predicted / energy,
                correlation=cross / float(np.sqrt(predicted * energy)) if predicted > 0 else 0.,
                full_native_energy_ratio=full_predicted / full_energy,
                full_native_correlation=(full_cross / float(np.sqrt(full_predicted * full_energy))
                                         if full_predicted > 0 else 0.))


def adam_regime(checkpoint):
    state = checkpoint.get('optimizer_state_dict') or {}
    out = []
    for group in state.get('param_groups', []):
        eps, beta2 = group['eps'], group['betas'][1]
        ratios = []
        for index in group['params']:
            s = state['state'].get(index) if isinstance(state.get('state'), dict) else None
            if not s or 'exp_avg_sq' not in s:
                continue
            step = float(s['step'])
            v = s['exp_avg_sq'].double() / (1 - beta2 ** step)
            nonzero = v[v > 0]
            if nonzero.numel():
                ratios.append(nonzero.sqrt().median().item() / eps)
        out.append(dict(lr=group['lr'], eps=eps, median_sqrt_v_over_eps=ratios))
    return out


def sector_curvature(head, readout, ds, view, polarization, mean_power, project, iterations):
    """Power iteration for lambda_max of one update's data-loss Hessian in (w_re, w_im).

    The trainer's update loss is sum over the sector's pulses of mean(|yhat - y|^2) / P, divided by the
    pulse count, with yhat linear in the coefficients; so H v = (2 / (count P)) grad_w sum_pulses
    Re<J v, yhat(w)> / n_pulse. Coefficients are restored afterwards.
    """
    field = head.field
    obs = list(ds.observations(*view, polarization))
    rs = [readout.for_observation(o) for o in obs]

    def render(o, r):
        x = head(o, r)[0]
        return RangeReadout.project(x, r) if project else x

    saved = field.w_re.detach().clone(), field.w_im.detach().clone()
    generator = torch.Generator().manual_seed(0)
    v = [torch.randn(w.shape, generator=generator).to(w) for w in (field.w_re, field.w_im)]
    estimates = []
    try:
        for _ in range(iterations):
            hv = [torch.zeros_like(w, dtype=torch.float64) for w in (field.w_re, field.w_im)]
            for o, r in zip(obs, rs):  # one pulse at a time: the Hessian-vector product is additive
                with torch.no_grad():
                    field.w_re.copy_(v[0])
                    field.w_im.copy_(v[1])
                    jv = render(o, r)
                    field.w_re.copy_(saved[0])
                    field.w_im.copy_(saved[1])
                out = render(o, r)
                f = (jv.conj() * out).real.sum() / out.numel()
                for acc, g in zip(hv, torch.autograd.grad(f, (field.w_re, field.w_im))):
                    acc += 2 * g.double() / (len(obs) * mean_power)
            norm_v = float(sum(x.double().square().sum() for x in v).sqrt())
            norm_hv = float(sum(x.square().sum() for x in hv).sqrt())
            estimates.append(norm_hv / norm_v)
            v = [(x / norm_hv).to(field.w_re) for x in hv]
    finally:
        with torch.no_grad():
            field.w_re.copy_(saved[0])
            field.w_im.copy_(saved[1])
    return dict(view=list(view), pulses=len(obs), lambda_max=estimates[-1],
                last_iterations=estimates[-3:], converged=abs(estimates[-1] - estimates[-2]) <= 1e-3 * estimates[-1])


def curvature_report(checkpoint, head, readout, ds, polarization, views, iterations):
    recipe = checkpoint['recipe']
    project = nufft.loss_domain(recipe) != nufft.FULL
    mean_power = checkpoint['training_statistics'][polarization]['mean_power']
    group = checkpoint['optimizer_state_dict']['param_groups'][0]
    lr, eps = group['lr'], group['eps']
    start = (checkpoint.get('initialization') or {}).get(polarization) or {}
    gain = float(torch.exp(head.gain.log_mag.detach()))
    start_gain = (abs(complex(*start['warm_start_gain'])) / start['coefficient_gauge']
                  if 'warm_start_gain' in start and 'coefficient_gauge' in start else None)
    to_start = (start_gain / gain) ** 2 if start_gain else None
    field = head.field
    active = field.active_mask
    orders = field.order.detach().clone()
    rows = []
    for view in views:
        row = dict(checkpoint_orders=sector_curvature(head, readout, ds, view, polarization, mean_power, project, iterations))
        with torch.no_grad():
            field.order[active] = 0
        try:
            row['order_zero'] = sector_curvature(head, readout, ds, view, polarization, mean_power, project, iterations)
        finally:
            with torch.no_grad():
                field.order.copy_(orders)
        for key in ('checkpoint_orders', 'order_zero'):
            lam = row[key]['lambda_max']
            row[key].update(lr_over_eps_lambda=lr / eps * lam,
                            lr_over_eps_lambda_at_start_gain=(lr / eps * lam * to_start) if to_start else None)
        row['order_ratio'] = row['checkpoint_orders']['lambda_max'] / row['order_zero']['lambda_max']
        rows.append(row)
        print('curvature', row, flush=True)
    return dict(lr=lr, eps=eps, mean_power=mean_power, loss_domain='projected' if project else 'full_native',
                checkpoint_gain=gain, start_gain=start_gain, hessian_scale_start_over_checkpoint=to_start,
                max_order=int(orders[active].max()), iterations=iterations, sectors=rows,
                bounds=dict(single_step=2.0, ema_momentum_fixed_quadratic=2 * 1.9 / 0.1))


def scene_scale(checkpoint, polarization):
    """Coefficient size and L1 prior against the B787 references (scene collapse shows here first).

    Validation RelMSE near 1 cannot tell "fits nothing" from "predicts zero"; the coefficient scale can.
    """
    state = {k.split('.field.', 1)[1]: v for k, v in checkpoint['model_state_dict'].items()
             if k.startswith(f'{polarization}.field.')}
    active = state['active_mask']
    unlocked = state['basis_degree'][None, :] <= state['order'][active][:, None]
    norm = ((state['w_re'][active].double().square() + state['w_im'][active].double().square()) * unlocked).sum(-1).sqrt()
    history = checkpoint.get('history') or []
    l1 = ((history[-1].get('priors') or {}).get(polarization) or {}).get('l1') if history else None
    mu1, reference = RIFT_DATASET_REFERENCE['mu1'], RIFT_DATASET_REFERENCE['coefficient_norm']
    # Gain-weighted L1 mass sum g|w| against the start's (m1 = g * sum|w| / initial points at the gauge).
    gain = float(torch.exp(checkpoint['model_state_dict'][f'{polarization}.gain.log_mag'].double()))
    start = (checkpoint.get('initialization') or {}).get(polarization) or {}
    start_mass = start['m1'] * start['initial_points'] if 'm1' in start and 'initial_points' in start else None
    mass = gain * float(norm.sum())
    return dict(active_points=int(active.sum()), mean_coefficient_norm=float(norm.mean()),
                gain_magnitude=gain, gain_weighted_l1_mass=mass,
                amplitude_ratio_to_start=(mass / start_mass) if start_mass else None,
                start_gauge_norm=reference, norm_over_start=float(norm.mean()) / reference if reference else None,
                last_epoch_l1=l1, mu1=mu1, l1_over_mu1=(l1 / mu1) if (l1 is not None and mu1) else None,
                l1_by_epoch=[((h.get('priors') or {}).get(polarization) or {}).get('l1') for h in history])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--dataset-root', default=os.environ.get('GOTCHA_DATA_ROOT',
                   '/scratch/user/u.db364833/GOTCHA-CP_Combined/GOTCHA-CP_Combined'))
    p.add_argument('--polarization', default='hh')
    p.add_argument('--sectors', type=int, default=24, help='random TRAIN and VALIDATION sectors each (seed 0)')
    p.add_argument('--recency', type=int, default=0, help='also score the last/first N sectors of the fixed order')
    p.add_argument('--shard-root', type=Path, help='the checkpoint\'s shard root (filtered targets)')
    p.add_argument('--region-config', type=Path, help='region config for regions outside the catalogue')
    p.add_argument('--curvature', type=int, default=0, help='TRAIN sectors for the Hessian power iteration')
    p.add_argument('--curvature-iterations', type=int, default=15)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--output', type=Path)
    args = p.parse_args(argv)
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    recipe = with_legacy_keys(checkpoint['recipe'], checkpoint['recipe']['method'])
    ds = dataset_for(checkpoint, args.dataset_root, args.shard_root, args.region_config)
    head = ChannelField(recipe['method'], ds.region, recipe, 'cpu')
    prefix = f'{args.polarization}.'
    head.load_state_dict({k[len(prefix):]: v for k, v in checkpoint['model_state_dict'].items()
                          if k.startswith(prefix)}, strict=True)
    head.eval()
    readout = RangeReadout(ds.region, device='cpu')
    rng = random.Random(0)
    report = dict(checkpoint=str(args.checkpoint.resolve()), epoch=checkpoint.get('epoch'),
                  updates=checkpoint.get('updates'), recipe={k: recipe.get(k) for k in
                  ('optimizer', 'lr', 'pos_lr', 'adam_eps', 'epochs', 'range_model', 'initialization', 'priors')},
                  adam=adam_regime(checkpoint), scene=scene_scale(checkpoint, args.polarization))
    print('scene', report['scene'], flush=True)
    for role in ('train', 'validation'):
        views = rng.sample(ds.viewpoints(role), min(args.sectors, len(ds.viewpoints(role))))
        report[role] = pooled(score(head, readout, ds, views, args.polarization))
        print(role, report[role], flush=True)
    if args.recency:
        views = ds.viewpoints('train')
        order = np.random.Generator(np.random.PCG64(recipe['seed'])).permutation(len(views)).tolist()
        start = (checkpoint.get('initialization') or {}).get(args.polarization, {}).get('views')
        if start and [list(views[i]) for i in order[:len(start)]] != [list(v) for v in start]:
            raise ValueError('Reconstructed training order disagrees with the saved start views')
        for label, indices in (('last_trained', order[-args.recency:]), ('first_of_epoch', order[:args.recency])):
            rows = score(head, readout, ds, [views[i] for i in indices], args.polarization)
            report[label] = dict(pooled(rows), per_sector_correlation=[
                r['cross'] / float(np.sqrt(r['predicted'] * r['energy'])) if r['predicted'] > 0 else 0. for r in rows])
            print(label, report[label], flush=True)
    if args.curvature:
        views = random.Random(1).sample(ds.viewpoints('train'), min(args.curvature, len(ds.viewpoints('train'))))
        report['curvature'] = curvature_report(checkpoint, head, readout, ds, args.polarization, views,
                                               args.curvature_iterations)
    print('adam', report['adam'], flush=True)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
