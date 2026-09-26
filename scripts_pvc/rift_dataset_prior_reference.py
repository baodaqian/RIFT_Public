"""Measure the RIFT-dataset prior strengths in dimensionless form, at initialization.

Runs the unchanged production RIFT-dataset command (``train_rift_dataset.py``
planner -> ``train.py`` full-scale adaptive argv, via ``train_pvc``'s device
rebinding) up to the point where ``train_sar`` would start. That is, after the
production backprojection initialization. It then replays the training loop's
first-view gain warm start and stops before any optimizer step. Reported per object:

    sigma2 = mean per-sample training power in the loss domain (all TRAIN views)
    m1     = |g| * mean_i ||w_i||      over the N0 initial points  (L1 scale)
    m2     = |g|^2 * mean_i ||w_i||^2  over the N0 initial points  (SH-energy scale)
    mu1    = l1_weight * m1 / sigma2          (L1 penalty / mean power at init)
    mu2    = sh_degree_weight * m2 / sigma2   (SH weight in units of initial scene energy)

The GOTCHA RIFT recipe uses mu1/mu2 so its priors keep the same strength
relative to its own data and its own backprojection start. The checkpoint
root is a scratch directory. The selected role manifest is only read, and
only TRAIN responses are touched.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault('RIFT_ACCELERATOR', 'cpu')
os.environ.setdefault('WANDB_MODE', 'disabled')

import torch  # noqa: E402

import train_pvc  # noqa: E402
import train_rift_dataset as planner  # noqa: E402


class _Captured(Exception):
    pass


def replace(argv, flag, value):
    argv[argv.index(flag) + 1] = str(value)


def measure(name, dataset_root, output_root, scratch, num_train, num_tx, num_rx):
    plan = planner.make_plan(planner.parse_args([
        '--dataset-root', str(dataset_root), '--output-root', str(output_root), '--object', name,
        '--num-train', str(num_train), '--num-tx', str(num_tx), '--num-rx', str(num_rx), '--method', 'rift']))
    command = list(plan['plans'][0]['commands'][0])
    argv = command[next(i for i, part in enumerate(command) if str(part).endswith('train.py')) + 1:]
    manifest = Path(argv[argv.index('--npz-role-manifest') + 1])
    if not manifest.is_file():
        raise FileNotFoundError(f'{manifest}: prepare the selected manifest first (read-only here)')
    replace(argv, '--checkpoint-root', Path(scratch) / name)
    train = train_pvc.install()
    result = {}

    def capture(num_epochs, model, train_loader, validation_loader, criterion, optimizer, scheduler,
                device, num_freq_selected, checkpoint_path, w_1, w_2, **kw):
        gain, op_kwargs = kw['gain'], kw.get('op_kwargs') or {}
        n0 = int(model.active_mask.sum())
        # First-view gain warm start, exactly as train_sar's loop performs it.
        batch = next(iter(train_loader))
        freqs, dphi, dtheta, mag, phase, rx, tx = batch
        mag_cube, phase_cube = train.reshape_measured_cubes(mag, phase, device, kw['num_tx'], kw['num_rx'])
        freqs = freqs.squeeze(0).to(device)
        idx = train.select_freq_indices(freqs.shape[0], num_freq_selected, device)
        pos, weights = model.active_scatterers(dtheta.to(device), dphi.to(device))
        weights = train.apply_occlusion(model, weights, rx.squeeze(0).to(device), tx.squeeze(0).to(device),
                                        kw.get('occlusion'))
        with torch.no_grad():
            pred = train.range_forward_operator(
                freqs, train.get_kvector(freqs, train.cc), rx.squeeze(0).to(device), tx.squeeze(0).to(device),
                pos, weights, phase_sign=kw['phase_sign'], freq_indices=idx,
                compute_dtype=kw['compute_dtype'], **op_kwargs)
            gain.maybe_init_scale(pred, torch.polar(mag_cube[:, :, idx], phase_cube[:, :, idx]).permute(2, 0, 1))
            g = float(torch.exp(gain.log_mag))
            mask = model.active_mask.reshape(-1)
            sq = (model.w_re.reshape(-1, model.w_re.shape[-1])[mask].double() ** 2
                  + model.w_im.reshape(-1, model.w_im.shape[-1])[mask].double() ** 2).sum(-1)
            m1 = g * float(sq.sqrt().sum()) / n0
            m2 = g * g * float(sq.sum()) / n0
            power, count = 0.0, 0
            for b in train_loader:
                mc, _ = train.reshape_measured_cubes(b[3], b[4], device, kw['num_tx'], kw['num_rx'])
                selected = train.select_freq_indices(b[0].shape[-1], num_freq_selected, device)
                power += float(mc[:, :, selected].double().square().sum())
                count += mc[:, :, selected].numel()
        l1_weight, sh_weight = kw['l1_weight'], kw['sh_smooth_weight']
        sigma2 = power / count
        result.update(object=name, command=argv, train_views=len(train_loader), initial_points=n0,
                      gain_magnitude=g, sigma2=sigma2, m1=m1, m2=m2, l1_weight=l1_weight,
                      sh_degree_weight=sh_weight, regularizer_normalization=kw['regularizer_normalization'],
                      mu1=l1_weight * m1 / sigma2, mu2=sh_weight * m2 / sigma2)
        raise _Captured

    train.train_sar = capture
    try:
        train.main(argv)
    except _Captured:
        return result
    raise RuntimeError('train.main returned without reaching train_sar')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True,
                   help='Planner output root holding the prepared selected role manifests (read only)')
    p.add_argument('--objects', nargs='+', default=['b787'])
    p.add_argument('--num-train', type=int, default=2400)
    p.add_argument('--num-tx', type=int, default=1)
    p.add_argument('--num-rx', type=int, default=1)
    p.add_argument('--out', type=Path)
    args = p.parse_args(argv)
    with tempfile.TemporaryDirectory() as scratch:
        results = [measure(name, args.dataset_root, args.output_root, scratch, args.num_train, args.num_tx,
                           args.num_rx) for name in args.objects]
    text = json.dumps(results, indent=2)
    print(text, flush=True)
    if args.out:
        args.out.write_text(text + '\n')


if __name__ == '__main__':
    main()
