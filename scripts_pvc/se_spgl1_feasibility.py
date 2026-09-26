"""Is the SE Eq. 4 residual budget reachable on a sub-aperture? LSQR on the SPGL1 lane's dense operator (diagnostic).

SPGL1 (``rift_pvc/sugavanam_ertin_spgl1.py``) solves min ||x||_1 s.t. ||A x - b|| <= sigma with
sigma = sqrt(2 N target_loss) (the recipe's 1% residual energy). If the least-squares floor
min ||A x - b|| lies above sigma, the constraint cannot be met and SPGL1 raises tau without bound
until an iteration limit. Plain LSQR (scipy, no regularization) on the same operator reports the
residual it reaches after k iterations: once below sigma the budget is reachable; a residual that
plateaus above sigma says it is not (at this k). Reads TRAIN rows only; writes a JSON report and
changes nothing in any run.

    python scripts_pvc/se_spgl1_feasibility.py --output OUT.json --groups 0,1 --iterations 1000 \\
        --gotcha <train_gotcha_dataset_pvc.py arguments>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.sparse.linalg import lsqr  # noqa: E402

from rift.sugavanam_ertin_acquisition import GOTCHAAcquisition, training_statistics  # noqa: E402
from rift.sugavanam_ertin_paper_workflow import grid_points, plan  # noqa: E402
from rift_pvc import sugavanam_ertin_spgl1 as se  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--groups', required=True)
    p.add_argument('--iterations', type=int, default=1000)
    p.add_argument('--checkpoints', default='200', help='extra iteration counts to report (comma list), each an independent LSQR run')
    p.add_argument('--gotcha', nargs=argparse.REMAINDER, required=True)
    args = p.parse_args(argv)
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', torch.get_num_threads())))
    import train_gotcha_dataset_pvc as frontend
    dataset, planned = frontend.make_plan(frontend.parse_args(args.gotcha))
    acquisition = GOTCHAAcquisition(dataset)
    recipe = se.make_recipe(acquisition.kind, planned['plans'][0]['config'])
    partition, _, planning = plan(acquisition, recipe)
    recipe = planning['recipe']
    statistics = training_statistics(acquisition)
    points = grid_points(acquisition.extent, recipe['granularity'], 'cpu')
    report = dict(schema='se_spgl1_feasibility_v1', gotcha=args.gotcha, iterations=args.iterations, groups={})
    for group in [int(g) for g in args.groups.split(',')]:
        started = time.perf_counter()
        ids = np.flatnonzero(partition.assignments == group).tolist()
        samples = sum(statistics['per_view_samples'][i] for i in ids)
        sigma = se.sigma_from_target(se.group_target(statistics, recipe, ids), samples)
        A, b = se.dense_operator(acquisition, points, ids, statistics)
        operator = se.linear_operator(A)
        bn = b.numpy()
        trace = {}

        def residual_at(k):
            x = lsqr(operator, bn, atol=0, btol=0, conlim=0, iter_lim=k)[0]
            return float(np.linalg.norm(A.numpy() @ x - bn))

        for k in sorted({int(c) for c in args.checkpoints.split(',') if c} | {args.iterations}):
            trace[k] = residual_at(k)
            print(json.dumps(dict(group=group, rows=A.shape[0], iterations=k, residual=trace[k], sigma=sigma,
                                  b_norm=float(np.linalg.norm(bn)))), flush=True)
        report['groups'][group] = dict(rows=int(A.shape[0]), views=len(ids), sigma=sigma, b_norm=float(np.linalg.norm(bn)),
                                       lsqr_residual=trace, reachable=trace[args.iterations] <= sigma,
                                       seconds=time.perf_counter() - started)
        del A, b, operator
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()
