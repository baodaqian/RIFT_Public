"""SE paper-v1 with the SPGL1 Stage-1 solver on a RIFT collection scene or GOTCHA (PVC lane).

    solve     --groups 0-35   one SPGL1 spg_bpdn solve per sub-aperture (CPU, dense Eq. 2 operator)
    assemble                  terminal Stage-1 checkpoint in the original paper-v1 schema
    stage2    [--stage1-only] the unchanged workflow resumed from checkpoint_latest.pt (XPU)

Source, one of:
    --npz-path NPZ --parent-role-manifest JSON --config JSON     RIFT collection scene
    --gotcha <train_gotcha_dataset_pvc.py arguments>              GOTCHA (must be last)
The GOTCHA arguments are parsed and planned by the PVC frontend itself
(``parse_args`` + ``make_plan``, method ``sugavanam_ertin`` alone), so the dataset,
pulse/frequency selection and SE configuration are exactly the frontend's; only
the output directory is this command's ``--output``.

Every command rebuilds the acquisition and the recipe via
``rift_pvc.sugavanam_ertin_spgl1.make_recipe``; all sub-aperture results and
checkpoints are bound to that identity. Training statistics are computed once and
kept in ``stage1_spgl1/statistics.pt``.
Exit codes follow the paper-v1 trainer: 0 complete / stage1 complete, 2 incomplete, 75 interrupted.
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

import torch  # noqa: E402

from rift.sugavanam_ertin_acquisition import (CollectionAcquisition, GOTCHAAcquisition, digest,  # noqa: E402
                                              training_statistics)
from rift_pvc import sugavanam_ertin_spgl1 as se_spgl1  # noqa: E402


def groups_arg(text):
    groups = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        groups.extend(range(int(a), int(b or a)+1))
    return groups


def statistics_for(acquisition, output):
    path = Path(output)/se_spgl1.GROUPS_DIR/"statistics.pt"
    if path.exists():
        statistics = torch.load(path, map_location="cpu", weights_only=False)
        if statistics["identity"] != digest(acquisition.identity):
            raise ValueError(f"{path} belongs to another acquisition")
        return statistics
    # First writer wins: parallel first jobs on different nodes must not each bind
    # their own (possibly last-bit different) statistics.
    se_spgl1.exclusive_save(training_statistics(acquisition), path)
    return statistics_for(acquisition, output)


def source(args, parser):
    """(acquisition, SE configuration) from exactly one of the two source specifications."""
    collection = (args.npz_path, args.parent_role_manifest, args.config)
    if args.gotcha is not None:
        if any(collection):
            parser.error("--gotcha replaces --npz-path/--parent-role-manifest/--config")
        import train_gotcha_dataset_pvc as frontend
        dataset, plan = frontend.make_plan(frontend.parse_args(args.gotcha))
        if [p["method"] for p in plan["plans"]] != ["sugavanam_ertin"]:
            parser.error("--gotcha must select --method sugavanam_ertin alone")
        return GOTCHAAcquisition(dataset), plan["plans"][0]["config"]
    if not all(collection):
        parser.error("a collection scene needs --npz-path, --parent-role-manifest and --config")
    return (CollectionAcquisition(npz_path=args.npz_path, manifest=args.parent_role_manifest),
            json.loads(args.config.read_text()))


def log(record):
    print(json.dumps(record, default=float), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["solve", "assemble", "stage2"])
    parser.add_argument("--npz-path")
    parser.add_argument("--parent-role-manifest")
    parser.add_argument("--config", type=Path, help="paper-v1 JSON overrides (e.g. se_g40_readout48.json)")
    parser.add_argument("--output", required=True, type=Path, help="SE output directory of this scene")
    parser.add_argument("--groups", type=groups_arg, help="solve: sub-apertures, e.g. 0-35 or 0,5,9")
    parser.add_argument("--verbosity", type=int, default=1, help="SPGL1 log level only")
    parser.add_argument("--stage1-only", action="store_true")
    parser.add_argument("--device", default=None, help="stage2 device; defaults to the active accelerator")
    parser.add_argument("--gotcha", nargs=argparse.REMAINDER,
                        help="GOTCHA source: the remaining arguments go to train_gotcha_dataset_pvc.py")
    args = parser.parse_args(argv)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", torch.get_num_threads())))
    acquisition, config = source(args, parser)
    recipe = se_spgl1.make_recipe(acquisition.kind, config)
    started = time.perf_counter()
    if args.command == "solve":
        if not args.groups:
            parser.error("solve requires --groups")
        statistics = statistics_for(acquisition, args.output)
        log(dict(event="setup", solver=se_spgl1.SOLVER, spgl1_commit=se_spgl1.SPGL1_COMMIT,
                 groups=args.groups, threads=torch.get_num_threads(), output=str(args.output)))
        se_spgl1.solve_groups(acquisition, recipe, statistics, args.output, args.groups,
                              verbosity=args.verbosity, log=log)
        log(dict(event="solve_done", seconds=time.perf_counter()-started))
        return 0
    if args.command == "assemble":
        statistics = statistics_for(acquisition, args.output)
        report = se_spgl1.assemble(acquisition, recipe, statistics, args.output)
        log(dict(event="assembled", seconds=time.perf_counter()-started, **report))
        return 0 if report["converged"] == report["groups"] else 2
    from rift.sugavanam_ertin_stage2_runtime_v1 import install_stop_handlers, stop_requested, reset_stop_request
    from rift_pvc import sugavanam_ertin_paper_workflow as workflow
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Real SE fitting requires an experiment-manager Slurm allocation")
    # The unchanged workflow recomputes the partition on this node and demands bit
    # equality with the checkpoint's; say so plainly before it refuses.
    from rift.sugavanam_ertin_paper_workflow import plan
    saved = torch.load(args.output/"checkpoint_latest.pt", map_location="cpu", weights_only=False)
    here = plan(acquisition, recipe)[0].record()
    if saved["partition"] != here:
        log(dict(event="stage2_partition_mismatch", node=os.uname().nodename,
                 **se_spgl1.partition_difference(saved["partition"], here),
                 reason="this node's numpy/libm does not reproduce the checkpoint's partition bits; "
                        "the original workflow requires equality, so rerun Stage 2 on another node"))
        return 2
    del saved
    reset_stop_request()
    install_stop_handlers()
    result = workflow.run(acquisition, recipe, args.output, device=workflow.resolve_device(args.device),
                          resume=args.output/"checkpoint_latest.pt", should_stop=stop_requested,
                          stage1_only=args.stage1_only)
    log(dict(event="stage2_result", seconds=time.perf_counter()-started, **result))
    return 75 if result["status"] == "interrupted" else (0 if result["status"] in ("complete", "stage1_complete") else 2)


if __name__ == "__main__":
    sys.exit(main())
