#!/usr/bin/env python
"""Model-free novel-view-synthesis bound for the radar view-sphere.

RIFT's headline metric is held-out complex rel-MSE.  Every published number so
far is produced by a *scene* model rendered through the forward operator, so we
have never measured how predictable the held-out signal is **without any scene
at all**.  That number is the floor any reviewer will ask for, and it separates
two very different stories:

  * if a dumb band-limited interpolator reaches a small error, the ~25% RIFT
    floor is MODEL error (the scene/operator cannot express the field) and there
    is real headroom;
  * if the interpolator also stalls near 25%, the held-out signal simply is not
    that predictable from 1,800 views at this operating point, and 25% is close
    to the information-theoretic bound.

Method.  For a fixed (tx, rx, frequency) triple the measured response is a
function S(u) on the view sphere.  With the target at the origin and the array
on a 10 m sphere, R_tx + R_rx ~= const - u.x, so S(u) is band-limited with
degree L ~= 2*k_max*R_target (for D = 0.10 m at 8.5-11.5 GHz that is L ~= 25).
We therefore fit a real spherical-harmonic expansion of degree L to the TRAIN
views only, evaluate it at the held-out views, and sweep L.  No scene, no
operator, no physics beyond "the field is band-limited on the view sphere".

The split is reproduced exactly from ``rift.npz_dataset.build_npz_dataloaders``
(seed-fixed permutation, ``--val-from-tail``), so the reported error is directly
comparable to the training runs' validation rel-MSE.

    module load anaconda3 && conda activate RIFT
    python scripts/eval_bandlimit_oracle.py \
        --npz-path data/b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere2k.npz \
        --num-train 1800 --num-val 200 --val-from-tail --seed 42
"""
import argparse
import json
import os

import numpy as np


def real_sh_basis(unit_vectors, max_degree):
    """Real spherical harmonics up to ``max_degree`` at ``unit_vectors`` (N,3).

    Returns ``(N, (max_degree+1)**2)`` float64.  Uses scipy's complex ``sph_harm``
    and the standard real combination, which is an orthonormal real basis for the
    same subspace -- the fit is basis-independent, so only the span matters.
    """
    from scipy.special import sph_harm

    x, y, z = unit_vectors[:, 0], unit_vectors[:, 1], unit_vectors[:, 2]
    theta = np.arccos(np.clip(z, -1.0, 1.0))          # polar, [0, pi]
    phi = np.arctan2(y, x)                            # azimuth, (-pi, pi]

    columns = []
    for l in range(max_degree + 1):
        for m in range(-l, l + 1):
            if m == 0:
                columns.append(np.real(sph_harm(0, l, phi, theta)))
            elif m > 0:
                ylm = sph_harm(m, l, phi, theta)
                columns.append(np.sqrt(2.0) * (-1.0) ** m * np.real(ylm))
            else:
                ylm = sph_harm(-m, l, phi, theta)
                columns.append(np.sqrt(2.0) * (-1.0) ** m * np.imag(ylm))
    return np.stack(columns, axis=1)


def split_indices(n_view, num_train, num_val, num_test, seed, val_from_tail):
    """Byte-for-byte the split in rift/npz_dataset.py::build_npz_dataloaders."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_view)
    if val_from_tail:
        val_idx = perm[n_view - num_val:]
        train_idx = perm[:num_train]
    else:
        train_idx = perm[:num_train]
        val_idx = perm[num_train:num_train + num_val]
    return train_idx, val_idx


def load_response_subset(npz_path, freq_stride, pair_stride):
    """Read the response cube, subsampling the frequency and Tx/Rx axes.

    The reported metric is a global sum over all pairs and frequencies, so a
    uniform subsample is an unbiased estimate of it.  Chirps are averaged away
    exactly as ``PecSphereNPZDataset`` does.
    """
    with np.load(npz_path, mmap_mode="r") as handle:
        response = handle["response"]                 # (V, Tx, Rx, chirp, F)
        positions = np.asarray(handle["viewpoint_positions"], dtype=np.float64)
        n_view, n_tx, n_rx, _, n_freq = response.shape
        freq_sel = np.arange(0, n_freq, freq_stride)
        tx_sel = np.arange(0, n_tx, pair_stride)
        rx_sel = np.arange(0, n_rx, pair_stride)
        block = np.empty((n_view, tx_sel.size * rx_sel.size * freq_sel.size),
                         dtype=np.complex64)
        for start in range(0, n_view, 64):
            stop = min(start + 64, n_view)
            chunk = np.asarray(response[start:stop])
            chunk = chunk.mean(axis=3)                # collapse the chirp axis
            chunk = chunk[:, tx_sel][:, :, rx_sel][:, :, :, freq_sel]
            block[start:stop] = chunk.reshape(stop - start, -1)
    return block, positions, (tx_sel.size, rx_sel.size, freq_sel.size)


def fit_and_score(basis_train, basis_val, y_train, y_val, ridge):
    """Least-squares SH fit on train views, global complex rel-MSE on val."""
    gram = basis_train.T @ basis_train
    if ridge > 0:
        gram = gram + ridge * np.trace(gram) / gram.shape[0] * np.eye(gram.shape[0])
    rhs = basis_train.T @ y_train
    coeffs = np.linalg.solve(gram, rhs)

    pred_val = basis_val @ coeffs
    val_err = np.sum(np.abs(pred_val - y_val) ** 2) / np.sum(np.abs(y_val) ** 2)

    pred_train = basis_train @ coeffs
    train_err = np.sum(np.abs(pred_train - y_train) ** 2) / np.sum(np.abs(y_train) ** 2)
    return float(train_err), float(val_err)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz-path", required=True)
    parser.add_argument("--num-train", type=int, default=1800)
    parser.add_argument("--num-val", type=int, default=200)
    parser.add_argument("--num-test", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-from-tail", action="store_true")
    parser.add_argument("--freq-stride", type=int, default=10,
                        help="Keep every Nth frequency (unbiased subsample of the global metric)")
    parser.add_argument("--pair-stride", type=int, default=2,
                        help="Keep every Nth Tx and Rx element")
    parser.add_argument("--max-degrees", type=int, nargs="+",
                        default=[0, 4, 8, 12, 16, 20, 24, 28, 32],
                        help="Sweep of SH bandwidths to fit on the view sphere")
    parser.add_argument("--ridge", type=float, default=0.0,
                        help="Relative Tikhonov weight on the normal equations")
    parser.add_argument("--train-sweep", type=int, nargs="*", default=None,
                        help="Also sweep the number of training views at the best degree")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    block, positions, shape = load_response_subset(
        args.npz_path, args.freq_stride, args.pair_stride)
    n_view = block.shape[0]
    print(f"loaded {args.npz_path}")
    print(f"  views={n_view}  kept (tx,rx,freq)={shape}  columns={block.shape[1]}")

    unit = positions / np.linalg.norm(positions, axis=1, keepdims=True)
    train_idx, val_idx = split_indices(
        n_view, args.num_train, args.num_val, args.num_test, args.seed, args.val_from_tail)
    print(f"  train={train_idx.size}  val={val_idx.size}  val_from_tail={args.val_from_tail}")

    y_train = block[train_idx].astype(np.complex128)
    y_val = block[val_idx].astype(np.complex128)

    # A constant predictor (the mean training response) is the trivial reference:
    # any method must beat this to be doing anything at all.
    baseline = np.sum(np.abs(y_train.mean(axis=0)[None, :] - y_val) ** 2) / np.sum(np.abs(y_val) ** 2)
    zero = 1.0
    print(f"\n  reference: predict-zero {100 * zero:.2f}%   predict-train-mean {100 * baseline:.2f}%")

    max_needed = max(args.max_degrees)
    full_basis = real_sh_basis(unit, max_needed)

    results = []
    print(f"\n{'degree':>7} {'n_coef':>7} {'train rel-MSE':>15} {'val rel-MSE':>13}")
    for degree in args.max_degrees:
        n_coef = (degree + 1) ** 2
        if n_coef >= train_idx.size:
            print(f"{degree:>7} {n_coef:>7}   skipped (n_coef >= n_train)")
            continue
        basis = full_basis[:, :n_coef]
        train_err, val_err = fit_and_score(
            basis[train_idx], basis[val_idx], y_train, y_val, args.ridge)
        results.append({"degree": degree, "n_coef": n_coef,
                        "train_rel_mse": train_err, "val_rel_mse": val_err})
        print(f"{degree:>7} {n_coef:>7} {100 * train_err:>14.4f}% {100 * val_err:>12.4f}%")

    best = min(results, key=lambda r: r["val_rel_mse"])
    print(f"\nBEST model-free bound: degree {best['degree']} "
          f"({best['n_coef']} coefficients per (tx,rx,freq)) "
          f"-> val rel-MSE {100 * best['val_rel_mse']:.4f}%")

    sweep = []
    if args.train_sweep:
        print(f"\ntraining-view sweep at degree {best['degree']}:")
        print(f"{'n_train':>8} {'val rel-MSE':>13}")
        basis = full_basis[:, :best["n_coef"]]
        for n_train in args.train_sweep:
            if n_train <= best["n_coef"]:
                print(f"{n_train:>8}   skipped (n_train <= n_coef)")
                continue
            subset = train_idx[:n_train]
            _, val_err = fit_and_score(
                basis[subset], basis[val_idx], block[subset].astype(np.complex128),
                y_val, args.ridge)
            sweep.append({"n_train": n_train, "val_rel_mse": val_err})
            print(f"{n_train:>8} {100 * val_err:>12.4f}%")

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as handle:
            json.dump({"npz_path": args.npz_path, "num_train": args.num_train,
                       "num_val": args.num_val, "seed": args.seed,
                       "val_from_tail": args.val_from_tail,
                       "freq_stride": args.freq_stride, "pair_stride": args.pair_stride,
                       "predict_zero": zero, "predict_train_mean": baseline,
                       "degree_sweep": results, "best": best,
                       "train_view_sweep": sweep}, handle, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
