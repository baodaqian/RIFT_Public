#!/usr/bin/env python
"""Honest model-free NVS bound as a function of training-view count, plus an
angular-EXTRAPOLATION split.

Why this exists
---------------
``scripts/eval_bandlimit_oracle.py`` measured the scene-free interpolation bound
at the full 1,800 training views (26.36%) and swept the view count.  That sweep
holds the SH degree fixed at 24 -- the degree chosen at n=1800 -- for every rung,
so its low-view numbers (n=800 -> 313%) are a MIS-SPECIFIED fit, not a bound:
625 coefficients fitted on 800 views nearly interpolates the training data and
blows up off it.  Comparing RIFT against that is comparing against a straw man.

This script re-selects the interpolator's capacity AT EVERY RUNG -- sweeping the
SH degree and a ridge grid and keeping the best -- which is the honest bound.
Selection is done on the held-out views themselves, so the baseline is given an
advantage no real method has; any RIFT margin measured against it is therefore
CONSERVATIVE.  That is deliberate: the claim we want to defend is that tying all
153,600 channels to one 3D scene beats angular interpolation, and it has to hold
against the strongest interpolator we can build, not the most convenient one.

Two regimes are measured:

``--mode ladder``
    Nested training subsets of the seed-fixed permutation against the FIXED
    tail validation set, exactly as ``--val-from-tail`` does in training, so
    each rung is directly comparable to a training run at that ``--num-train``.

``--mode cap``
    Angular extrapolation.  The held-out views are a polar CAP (the views whose
    look direction is closest to an axis), so validation lies outside the convex
    hull of the training directions.  A band-limited SH fit is an interpolator;
    extrapolating one past the sampled region is ill-posed, whereas the forward
    operator is constrained by physics everywhere.  A random split of the same
    train/val sizes is reported alongside, so the extrapolation penalty is
    isolated from the change in view count.

    module load anaconda3 && conda activate RIFT
    python scripts/eval_nvs_view_ladder.py --mode ladder \
        --npz-path data/b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere2k.npz
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_bandlimit_oracle import (  # noqa: E402  (frozen module, imported not edited)
    load_response_subset,
    real_sh_basis,
    split_indices,
)


def best_fit_over_capacity(basis_train, basis_val, y_train, y_val, degrees, ridges,
                           max_coef_fraction):
    """Sweep (degree, ridge) and return the best val rel-MSE and the full grid.

    The Gram matrix and right-hand side are built ONCE at the largest admissible
    degree; the leading ``n_coef`` block of each is exactly the Gram/rhs of the
    first ``n_coef`` basis columns, so every smaller degree is a free slice.
    """
    n_train = basis_train.shape[0]
    # A fit whose coefficient count approaches the view count interpolates the
    # training views and says nothing about held-out ones.  Cap the capacity at
    # a fraction of the available views and report what that cap was.
    coef_budget = max(1, int(max_coef_fraction * n_train))
    admissible = [d for d in degrees if (d + 1) ** 2 <= min(coef_budget, n_train - 1)]
    if not admissible:
        return None, []

    n_max = (max(admissible) + 1) ** 2
    bt = basis_train[:, :n_max]
    gram_full = bt.T @ bt
    rhs_full = bt.T @ y_train
    denom_val = np.sum(np.abs(y_val) ** 2)
    denom_train = np.sum(np.abs(y_train) ** 2)

    grid = []
    for degree in admissible:
        n_coef = (degree + 1) ** 2
        gram = gram_full[:n_coef, :n_coef]
        rhs = rhs_full[:n_coef]
        scale = np.trace(gram) / n_coef
        for ridge in ridges:
            lhs = gram if ridge == 0 else gram + ridge * scale * np.eye(n_coef)
            try:
                coeffs = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                continue
            pred_val = basis_val[:, :n_coef] @ coeffs
            pred_train = bt[:, :n_coef] @ coeffs
            grid.append({
                "degree": degree,
                "n_coef": n_coef,
                "ridge": ridge,
                "train_rel_mse": float(np.sum(np.abs(pred_train - y_train) ** 2) / denom_train),
                "val_rel_mse": float(np.sum(np.abs(pred_val - y_val) ** 2) / denom_val),
            })
    if not grid:
        return None, []
    return min(grid, key=lambda r: r["val_rel_mse"]), grid


def polar_cap_split(unit_vectors, num_val, axis):
    """Hold out the ``num_val`` views closest to ``axis`` as a polar cap.

    Returns (train_idx, val_idx).  Validation directions then lie OUTSIDE the
    convex hull of the training directions, which is what makes this an
    extrapolation test rather than an interpolation one.
    """
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    projection = unit_vectors @ axis
    order = np.argsort(-projection)          # most-aligned first
    val_idx = order[:num_val]
    train_idx = order[num_val:]
    half_angle = float(np.degrees(np.arccos(np.clip(projection[val_idx].min(), -1.0, 1.0))))
    return train_idx, val_idx, half_angle


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz-path", required=True)
    p.add_argument("--mode", choices=["ladder", "cap"], default="ladder")
    p.add_argument("--num-train", type=int, default=1800)
    p.add_argument("--num-val", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--freq-stride", type=int, default=20)
    p.add_argument("--pair-stride", type=int, default=4)
    p.add_argument("--ladder", type=int, nargs="+",
                   default=[100, 200, 400, 800, 1200, 1600, 1800],
                   help="training-view counts to evaluate (nested subsets)")
    p.add_argument("--degrees", type=int, nargs="+",
                   default=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 32],
                   help="SH degrees to try at each rung")
    p.add_argument("--ridges", type=float, nargs="+",
                   default=[0.0, 1e-6, 1e-4, 1e-2, 1e-1],
                   help="ridge grid (relative to mean Gram diagonal) at each rung")
    p.add_argument("--max-coef-fraction", type=float, default=0.5,
                   help="cap coefficients at this fraction of the rung's view count")
    p.add_argument("--cap-axis", type=float, nargs=3, default=[0.0, 0.0, 1.0])
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    block, positions, shape = load_response_subset(
        args.npz_path, args.freq_stride, args.pair_stride)
    n_view = block.shape[0]
    unit = positions / np.linalg.norm(positions, axis=1, keepdims=True)
    print(f"loaded {args.npz_path}")
    print(f"  views={n_view}  kept (tx,rx,freq)={shape}  columns={block.shape[1]}")

    max_degree = max(args.degrees)
    full_basis = real_sh_basis(unit, max_degree)
    payload = {"npz_path": args.npz_path, "mode": args.mode, "seed": args.seed,
               "freq_stride": args.freq_stride, "pair_stride": args.pair_stride,
               "max_coef_fraction": args.max_coef_fraction,
               "degrees": args.degrees, "ridges": args.ridges}

    if args.mode == "ladder":
        train_pool, val_idx = split_indices(
            n_view, args.num_train, args.num_val, 0, args.seed, val_from_tail=True)
        y_val = block[val_idx].astype(np.complex128)
        print(f"  fixed held-out tail: {val_idx.size} views (val_from_tail, seed {args.seed})")
        print("\n  capacity re-selected at EVERY rung; selection uses the held-out")
        print("  views, so the baseline is stronger than any real method could be.\n")
        print(f"{'n_train':>8} {'best deg':>9} {'n_coef':>7} {'ridge':>8} "
              f"{'train':>9} {'VAL rel-MSE':>13} {'fixed-deg24':>12}")

        rows = []
        for n_train in args.ladder:
            subset = train_pool[:n_train]
            y_train = block[subset].astype(np.complex128)
            best, grid = best_fit_over_capacity(
                full_basis[subset], full_basis[val_idx], y_train, y_val,
                args.degrees, args.ridges, args.max_coef_fraction)
            if best is None:
                print(f"{n_train:>8}   no admissible capacity")
                continue
            # The mis-specified comparison the previous sweep reported.
            fixed = [r for r in grid if r["degree"] == 24 and r["ridge"] == 0.0]
            fixed_txt = f"{100 * fixed[0]['val_rel_mse']:>11.2f}%" if fixed else f"{'n/a':>12}"
            print(f"{n_train:>8} {best['degree']:>9} {best['n_coef']:>7} "
                  f"{best['ridge']:>8.0e} {100 * best['train_rel_mse']:>8.2f}% "
                  f"{100 * best['val_rel_mse']:>12.2f}% {fixed_txt}")
            rows.append({"n_train": n_train, "best": best,
                         "fixed_degree24": fixed[0] if fixed else None})
        payload["ladder"] = rows

    else:
        train_idx, val_idx, half_angle = polar_cap_split(unit, args.num_val, args.cap_axis)
        y_train = block[train_idx].astype(np.complex128)
        y_val = block[val_idx].astype(np.complex128)
        print(f"  polar-cap split about axis {args.cap_axis}: "
              f"{val_idx.size} held-out views inside a {half_angle:.1f} deg cap, "
              f"{train_idx.size} training views outside it")
        cap_best, _ = best_fit_over_capacity(
            full_basis[train_idx], full_basis[val_idx], y_train, y_val,
            args.degrees, args.ridges, args.max_coef_fraction)

        # Matched random split: same train/val sizes, so the only difference is
        # WHERE the held-out views sit, not how many there are.
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(n_view)
        r_val, r_train = perm[:val_idx.size], perm[val_idx.size:val_idx.size + train_idx.size]
        rand_best, _ = best_fit_over_capacity(
            full_basis[r_train], full_basis[r_val],
            block[r_train].astype(np.complex128), block[r_val].astype(np.complex128),
            args.degrees, args.ridges, args.max_coef_fraction)

        print(f"\n{'split':>22} {'best deg':>9} {'ridge':>8} {'train':>9} {'VAL rel-MSE':>13}")
        for label, res in (("polar cap (EXTRAP)", cap_best), ("random (interp)", rand_best)):
            if res is None:
                print(f"{label:>22}   no admissible capacity")
                continue
            print(f"{label:>22} {res['degree']:>9} {res['ridge']:>8.0e} "
                  f"{100 * res['train_rel_mse']:>8.2f}% {100 * res['val_rel_mse']:>12.2f}%")
        payload["cap"] = {"half_angle_deg": half_angle, "num_val": int(val_idx.size),
                          "num_train": int(train_idx.size),
                          "polar_cap": cap_best, "matched_random": rand_best}

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
