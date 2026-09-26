#!/usr/bin/env python3
"""New-target step 4c: per-pass phase constants from leave-one-pass-out summaries (docs/RIFT_GOTCHA_Tune.md A73-A75).

The Camry's adopted constants (carphase_v2, A56/B60) were built in two rounds, and this repeats that recipe for a new
target from ``gotcha_target_pass_phase.py --loo`` / ``--summarize-loo`` outputs:

- round 1 (``--round1``, measured on the target's uncorrected box shards): phi1_p = the |rho_c|-weighted circular mean
  phase of pass p at LOO iteration ``--iteration`` (2, as B56);
- round 2 (``--round2``, measured on shards rotated by phi1): residual r_p the same way; the final constant is
  phi_p = phi1_p + alpha r_p with alpha = 0.55 (the damped update B58 advised and A56 applied).

Without ``--round2`` the output is round 1's constants. The file is ``gotcha_target_rotate_passphase.py
--offsets-json`` input: rows of pass p are multiplied by exp(-i phi_p).

    python scripts_pvc/gotcha_target_phase_constants.py --round1 r1_summary.json --tag sentra_r1 --output offsets_r1.json
    python scripts_pvc/gotcha_target_phase_constants.py --round1 r1_summary.json --round2 r2_summary.json \\
        --tag sentra_v2 --output offsets_v2.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def weighted(summary_path, iteration):
    summary = json.loads(Path(summary_path).read_text())
    per_pass = summary['by_iteration'][str(iteration)]['per_pass']
    return {p: float(v['weighted_mean_phase_deg']) for p, v in per_pass.items()}, summary


def wrap_deg(x):
    return (x + 180.0) % 360.0 - 180.0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--round1', type=Path, required=True)
    p.add_argument('--round2', type=Path)
    p.add_argument('--iteration', type=int, default=2)
    p.add_argument('--alpha', type=float, default=0.55)
    p.add_argument('--tag', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    phi1, _ = weighted(args.round1, args.iteration)
    record = dict(round1_deg={k: round(v, 3) for k, v in sorted(phi1.items(), key=lambda kv: int(kv[0]))})
    phi = dict(phi1)
    if args.round2 is not None:
        residual, _ = weighted(args.round2, args.iteration)
        if set(residual) != set(phi1):
            raise ValueError('round 1 and round 2 cover different passes')
        phi = {k: wrap_deg(phi1[k] + args.alpha * residual[k]) for k in phi1}
        record.update(round2_residual_deg={k: round(v, 3) for k, v in sorted(residual.items(), key=lambda kv: int(kv[0]))},
                      alpha=args.alpha)
    passes = sorted(phi, key=int)
    out = dict(offsets_rad={k: math.radians(phi[k]) for k in passes}, offsets_deg={k: round(phi[k], 2) for k in passes},
               **record, iteration=args.iteration,
               convention='stored rows of pass p multiplied by exp(-i phi_p); phi_p = pass p phase against the '
                          'leave-one-pass-out consensus at the target, |rho_c|-weighted circular mean over the '
                          'measured all-TRAIN sector IDs',
               source=f'scripts_pvc/gotcha_target_phase_constants.py from {args.round1}'
                      + (f' and {args.round2}' if args.round2 else ''),
               recipe='Camry carphase_v2 (A56/B58/B60): round-1 constants, then + alpha x round-2 residuals',
               tag=args.tag)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out['offsets_deg']))


if __name__ == '__main__':
    main()
