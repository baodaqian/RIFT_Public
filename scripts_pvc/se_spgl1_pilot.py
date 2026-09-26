"""Pilot: SE Stage-1 Eq. 4 on production sub-apertures with SPGL1 defaults (no campaign effect).

Rebuilds the production acquisition/recipe/partition, checks them against the
production checkpoint, builds the dense Eq. 2 operator of each requested
sub-aperture, checks it against the batched objective, runs ``spgl1.spg_bpdn``
with the package defaults and records SPGL1's exit status plus the
repository's Eq. 4 certificates as diagnostics. Writes group_XX.json/.pt.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from rift.sugavanam_ertin_acquisition import CollectionAcquisition, digest
from rift.sugavanam_ertin_paper_workflow import grid_points, make_recipe, plan
from rift_pvc import sugavanam_ertin_spgl1 as se_spgl1
from rift_pvc.sugavanam_ertin_batched import batched_data_objective


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz-path", required=True)
    parser.add_argument("--parent-role-manifest", required=True)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path, help="production Stage-1 checkpoint")
    parser.add_argument("--groups", required=True, help="comma-separated sub-aperture indices")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--verbosity", type=int, default=1, help="SPGL1 log level only")
    args = parser.parse_args(argv)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", torch.get_num_threads())))
    args.output.mkdir(parents=True, exist_ok=True)
    import spgl1
    source = Path(spgl1.__file__).resolve().parents[1]
    commit = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

    acquisition = CollectionAcquisition(npz_path=args.npz_path, manifest=args.parent_role_manifest)
    recipe = make_recipe(acquisition.kind, json.loads(args.config.read_text()))
    partition, _, planning = plan(acquisition, recipe)
    recipe = planning["recipe"]
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if saved["acquisition"] != acquisition.identity or saved["recipe"] != recipe \
            or saved["partition"] != partition.record():
        raise ValueError("Pilot acquisition/recipe/partition differ from the production checkpoint")
    statistics = saved["statistics"]
    if statistics["identity"] != digest(acquisition.identity):
        raise ValueError("Production statistics belong to another acquisition")
    points = grid_points(acquisition.extent, recipe["granularity"], "cpu")
    print(json.dumps(dict(event="setup", solver=se_spgl1.SOLVER, spgl1_repository=se_spgl1.SPGL1_REPOSITORY,
                          spgl1_commit=commit, threads=torch.get_num_threads())), flush=True)

    for group in (int(g) for g in args.groups.split(",")):
        indices = np.flatnonzero(partition.assignments == group).tolist()
        samples = sum(statistics["per_view_samples"][i] for i in indices)
        target = saved["sparse_solvers"][group]["target_loss"]
        started = time.perf_counter()
        A, b = se_spgl1.dense_operator(acquisition, points, indices, statistics)
        result = dict(group=group, views=len(indices), rows=A.shape[0], samples=samples, target_loss=target,
                      operator_seconds=time.perf_counter()-started, spgl1_repository=se_spgl1.SPGL1_REPOSITORY,
                      spgl1_commit=commit)
        checks = {}
        generator = torch.Generator().manual_seed(group)
        random = (torch.randn(len(points), generator=generator, dtype=torch.float64)
                  + 1j*torch.randn(len(points), generator=generator, dtype=torch.float64))*1e-3
        for name, x in (("random", random), ("production_terminal", saved["fields"][group])):
            reference, reference_grad = batched_data_objective(acquisition, points, x, indices, statistics,
                                                               recipe, gradient=True)
            residual = A @ x-b
            value = .5*float(residual.abs().square().sum())/samples
            grad = (A.T @ residual.conj()).conj()/samples
            checks[name] = dict(value_relative_difference=abs(value-reference)/reference,
                                gradient_relative_difference=float((grad-reference_grad).norm()/reference_grad.norm()))
        result["operator_checks"] = checks
        sigma = se_spgl1.sigma_from_target(target, samples)
        result.update(sigma=sigma, b_norm=float(b.norm()))
        print(json.dumps(dict(event="operator", **result)), flush=True)
        started = time.perf_counter()
        x, info = se_spgl1.solve(A, b, sigma, verbosity=args.verbosity)
        result["spgl1_seconds"] = time.perf_counter()-started
        result["spgl1_info"] = {k: (v.tolist() if isinstance(v, np.ndarray) and v.size < 2 else v)
                                for k, v in info.items() if not isinstance(v, np.ndarray) or v.size < 2}
        result["spgl1_trace"] = {k: info[k][::10].tolist() for k in ("xnorm1", "rnorm2", "lambdaa")}
        result["repository_diagnostics"] = se_spgl1.repository_diagnostics(A, b, x, float(info["tau"]), target, samples)
        torch.save(dict(group=group, field=torch.from_numpy(x), info=result["spgl1_info"]), args.output/f"group_{group:02d}.pt")
        (args.output/f"group_{group:02d}.json").write_text(json.dumps(result, indent=1, default=float))
        print(json.dumps(dict(event="group_done", group=group, stat=int(info["stat"]), niters=int(info["niters"]),
            nprodA=int(info["nprodA"]), nprodAt=int(info["nprodAt"]), n_newton=int(info["n_newton"]),
            rnorm=float(info["rnorm"]), sigma=sigma, tau=float(info["tau"]), seconds=result["spgl1_seconds"],
            diagnostics=result["repository_diagnostics"])), flush=True)
        del A, b


if __name__ == "__main__":
    main()
