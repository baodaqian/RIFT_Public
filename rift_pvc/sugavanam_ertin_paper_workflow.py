"""PVC (Intel XPU) twin of ``rift.sugavanam_ertin_paper_workflow``.

This is the Sugavanam--Ertin production lane: the RIFT and GOTCHA frontends
reach it through ``train_sugavanam_ertin_pvc.py --recipe paper-v1``.

Execution adaptation (2026-09-22, GOTCHA.md section 7): ``run`` evaluates
Stage 1 with the batched sub-aperture objective of
``rift_pvc.sugavanam_ertin_batched`` (same recipe, identity, partition and
checkpoint schema; values equal up to summation order) and hands every later
stage to the original workflow through a resume. The remainder of this
docstring describes the device-default adaptation that still applies.

The audit in RIFT_PVC_Adaptation.md (section C.1) found exactly **two** CUDA
touches in the original, both of them a ``"cuda"`` *default*:

    run(..., device="cuda", ...)                     L371
    p.add_argument("--device", default="cuda")       L600

Everything else -- the Eq. 4 sparse solver, the sub-aperture partition, the
Stage-1 cloud extraction, the SDF losses, the published-initialization gate,
the surface export and the checkpoint schema -- is already fully
device-parameterized. So this module **imports the original and overrides only
those two defaults**. No scientific code is duplicated, which is the point: the
recipe, the gate and the checkpoint contract cannot drift between the CUDA and
PVC lanes because there is only one copy of them.

Two properties of the original make the XPU port unusually clean, both
verified and recorded in the audit:

* **No device generator anywhere on this lane.** Every ``torch.Generator`` in
  the original (L262, L486, L507) and in ``sugavanam_ertin_paper`` (L192,
  L240) is a CPU generator by deliberate design -- "CPU generator gives
  identical sampling state independent of GPU count". The XPU JIT hang that
  dispatch rule 5 warns about cannot occur here. As a consequence
  ``state["generator_state"]`` is a CPU payload, so a paper-v1 checkpoint has
  the same RNG schema on PVC and on H100 and its sampling stream is
  trajectory-identical across backends.

* **The published-initialization gate is carried over untouched.** It is
  imported, not reimplemented, so it cannot be relaxed or bypassed from here.
  See ``initialization_gate_reference()`` for the gate-integrity check.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from rift import sugavanam_ertin_paper_workflow as _original
from rift.sugavanam_ertin_acquisition import array_digest, training_statistics
from rift.sugavanam_ertin_paper import PaperSDF, initialization_audit
from rift.sugavanam_ertin_sparse import constrained_step, initial_state
from rift_pvc.sugavanam_ertin_batched import (ELEMENT_BUDGET, EXECUTION, batched_data_objective,
                                              batched_validation_readout)
from rift.sugavanam_ertin_paper_workflow import (  # re-exported unchanged
    SCHEMA,
    extract_cloud,
    grid_points,
    make_recipe,
    plan,
    refresh_surface_cpu_generator,
    validate_resume,
    _model_config,
)
from rift.sugavanam_ertin_acquisition import CollectionAcquisition, GOTCHAAcquisition
from rift_pvc import accelerator

__all__ = ["run", "run_gotcha", "main", "resolve_device", "initialization_gate_reference",
           "SCHEMA", "make_recipe", "plan", "grid_points", "extract_cloud",
           "validate_resume", "refresh_surface_cpu_generator", "_model_config"]


def resolve_device(device=None) -> torch.device:
    """The device to run on.

    ``None`` (the PVC default) selects the active accelerator: ``xpu`` on a PVC
    card, ``cuda`` on an H100, ``cpu`` otherwise. An explicit value is honoured
    verbatim, so the frontends, the tests and ``--device`` keep the original
    behaviour and can still force ``cpu``.
    """
    return accelerator.device() if device is None else torch.device(device)


def run(acquisition, recipe, output, *, device=None, resume=None, should_stop=None,
        stage1_only=False):
    """Sparse Stage 1 with the batched sub-aperture objective, then the original workflow.

    Same signature, recipe, acquisition identity, partition and checkpoint
    schema as :func:`rift.sugavanam_ertin_paper_workflow.run`. The Stage-1 loop
    is a copy of the original's with ``data_objective`` and
    ``validation_readout`` replaced by their batched PVC twins
    (``rift_pvc.sugavanam_ertin_batched``; values equal up to floating-point
    summation order, disclosed in ``execution.json`` and in every Stage-1
    history row). Once Stage 1 has completed, or when a resumed checkpoint is
    already past Stage 1, the original ``run`` continues from
    ``checkpoint_latest.pt``: the convergence gate, the cloud extraction,
    Stage 2 and the surface export are the originals, not copies.
    """
    device = resolve_device(device)
    partition, validation_assignments, planning = plan(acquisition, recipe)
    recipe = planning["recipe"]
    root = Path(output).absolute()
    state = None
    if resume:
        if Path(resume).absolute().parent != root:
            raise ValueError("Resume must use its original output directory")
        state = validate_resume(_original._torch_load(resume), acquisition, recipe, partition)
        if stage1_only and state["phase"] not in ("stage1", "stage1_unconverged"):
            raise ValueError("Stage-1-only execution requires a Stage-1 checkpoint")
        if state["phase"] != "stage1":
            return _original.run(acquisition, recipe, root, device=device, resume=resume,
                                 should_stop=should_stop, stage1_only=stage1_only)
    elif root.exists() and any(root.iterdir()):
        raise FileExistsError("Use a fresh SE paper-v1 output directory or explicit resume")
    # Identity and progress have been checked before any real response read.
    points = grid_points(acquisition.extent, recipe["granularity"], device)
    groups = len(partition.directions)
    if state is None:
        probe_model = PaperSDF(**_model_config(recipe, acquisition.extent)).to(device)
        probe = initialization_audit(probe_model, acquisition.extent, seed=recipe["seed"])
        del probe_model
        if probe["status"] == "initialization_degenerate":
            root.mkdir(parents=True, exist_ok=True)
            report = dict(status=probe["status"], schema=SCHEMA, output=str(root),
                acquisition=acquisition.identity, recipe=recipe, initialization_audit=probe,
                benchmark_eligible=False, reason="Configured Gaussian initialization is numerically degenerate; author details unresolved")
            _original.atomic_json_dump(report, root/"initialization.json")
            _original.atomic_json_dump(report, root/"status.json")
            return report
        statistics = training_statistics(acquisition)
        state = dict(schema=SCHEMA, acquisition=acquisition.identity, recipe=recipe,
            partition=partition.record(), statistics=statistics, phase="stage1",
            fields=torch.zeros(groups, len(points), dtype=torch.complex128),
            sparse_solvers=[], iteration=0, group_cursor=0,
            view_exposures=[0]*len(acquisition.keys["train"]), best=None,
            stage1_history=[], sdf_history=[], iso_history=[])
        for group in range(groups):
            ids = np.flatnonzero(partition.assignments == group)
            target = .5*recipe["residual_relative_energy"]*sum(statistics["per_view_energy"][i] for i in ids)/(sum(statistics["per_view_samples"][i] for i in ids)*statistics["rms"]**2)
            state["sparse_solvers"].append(initial_state(target, recipe["initial_lipschitz"]))
    root.mkdir(parents=True, exist_ok=True)
    _original.atomic_json_dump(dict(planning, execution=EXECUTION), root/"recipe.json")
    _original.atomic_json_dump(dict(schema=SCHEMA, execution=EXECUTION, element_budget=ELEMENT_BUDGET,
        stage1="batched_subaperture_objective_and_validation_readout_same_recipe_identity",
        later_stages="original_workflow_resumed_from_checkpoint_latest"), root/"execution.json")
    if not resume:
        _original._save(state, root)  # Durable zero-update state before the first inverse solve.
    stop = should_stop or (lambda: False)
    while state["iteration"] < recipe["stage1_iterations"]:
        group = state["group_cursor"]
        indices = np.flatnonzero(partition.assignments == group).tolist()
        x = state["fields"][group].to(device)
        def value_grad(w):
            return batched_data_objective(acquisition, points, w, indices, state["statistics"], recipe, gradient=True)
        def value(w):
            return batched_data_objective(acquisition, points, w, indices, state["statistics"], recipe)
        x, solver, audit = constrained_step(x, value_grad, value, state["sparse_solvers"][group],
            residual_rtol=recipe["residual_rtol"], optimality_rtol=recipe["optimality_rtol"])
        state["fields"][group] = x.cpu()
        state["sparse_solvers"][group] = solver
        for i in indices:
            state["view_exposures"][i] += 1
        state["stage1_history"].append(dict(iteration=state["iteration"]+1, group=group,
            sparse_solver=dict(solver), fields_sha256=array_digest(x.cpu().numpy()), execution=EXECUTION, **audit))
        state["group_cursor"] = (group+1) % groups
        if state["group_cursor"] == 0:
            state["iteration"] += 1
            iteration = state["iteration"]
            if iteration % recipe["validation_every"] == 0 or iteration == recipe["stage1_iterations"]:
                metrics = batched_validation_readout(acquisition, points, state["fields"], validation_assignments,
                                                     state["statistics"], recipe)
                state["best"] = dict(iteration=iteration, fields=state["fields"].clone(), validation=metrics)
                print(json.dumps(dict(stage=1, iteration=iteration, **metrics)), flush=True)
        completed = state["iteration"]*groups + state["group_cursor"]
        stopping = stop()
        if completed % recipe["checkpoint_every"] == 0 or stopping or state["iteration"] == recipe["stage1_iterations"]:
            _original._save(state, root)
        if stopping:
            return dict(status="interrupted", phase="stage1", output=str(root))
    # Stage 1 complete and durable: the original workflow takes over from here.
    return _original.run(acquisition, recipe, root, device=device, resume=root/"checkpoint_latest.pt",
                         should_stop=should_stop, stage1_only=stage1_only)


def run_gotcha(*, dataset, output_dir, config, device, resume):
    """GOTCHA entry point imported by ``train_gotcha_dataset_pvc.py``.

    Copy of the original's with this module's ``run`` (batched Stage 1); the
    Slurm-allocation requirement and the failed-status escalation are kept.
    """
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Real GOTCHA SE fitting requires an experiment-manager Slurm allocation")
    from rift.sugavanam_ertin_stage2_runtime_v1 import install_stop_handlers, stop_requested, reset_stop_request
    acquisition = GOTCHAAcquisition(dataset)
    recipe = make_recipe(acquisition.kind, config)
    reset_stop_request()
    install_stop_handlers()
    result = run(acquisition, recipe, output_dir, device=resolve_device(device), resume=resume, should_stop=stop_requested)
    if result["status"] not in ("complete", "interrupted"):
        raise RuntimeError(f"SE comparison incomplete: {result['status']}; see {output_dir}/status.json")
    return result


def initialization_gate_reference(kind, extent, *, config=None):
    """Evaluate the published-initialization gate without fitting or radar reads.

    The gate itself is imported from ``rift.sugavanam_ertin_paper``; this only
    runs it and hands back the audit. Used by the PVC tests and by the
    gate-integrity smoke to show the check still fires identically on XPU:
    ``initialization_std`` 1.0 (the literal paper value) must stay
    ``initialization_degenerate``, the production 0.05 must pass.
    """
    recipe = make_recipe(kind, config)
    model = PaperSDF(**_model_config(recipe, extent))
    return recipe, initialization_audit(model, extent, seed=recipe["seed"])


def run_gotcha_stage1(*, dataset, output_dir, config, device, resume):
    """Stop at the original sparse-stage boundary without changing its recipe.

    A successful return requires the workflow's convergence gates and durable
    checkpoint_stage1_final.pt. The usual frontend resumes that checkpoint to
    enter Stage 2; failed initialization/convergence must fail the Slurm job.
    """
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Real GOTCHA SE fitting requires an experiment-manager Slurm allocation")
    from rift.sugavanam_ertin_stage2_runtime_v1 import install_stop_handlers, stop_requested, reset_stop_request
    acquisition = GOTCHAAcquisition(dataset)
    recipe = make_recipe(acquisition.kind, config)
    reset_stop_request()
    install_stop_handlers()
    result = run(acquisition, recipe, output_dir, device=device, resume=resume,
                 should_stop=stop_requested, stage1_only=True)
    if result["status"] not in ("stage1_complete", "interrupted"):
        raise RuntimeError(f"SE comparison incomplete: {result['status']}; see {output_dir}/status.json")
    return result


def build_parser() -> argparse.ArgumentParser:
    """The original CLI with ``--device`` defaulting to the active backend."""
    from rift.rift_dataset import DEFAULT_ROOT
    p = argparse.ArgumentParser(description=_original.__doc__)
    p.add_argument("--recipe", choices=["paper-v1"], default="paper-v1")
    p.add_argument("--object")
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--npz-path")
    p.add_argument("--parent-role-manifest")
    p.add_argument("--checkpoint-root", type=Path)
    p.add_argument("--config", type=Path, help="JSON overrides for the explicit SE paper recipe")
    p.add_argument("--resume", type=Path)
    p.add_argument("--stage1-only", action="store_true",
                   help="Stop after sparse Stage 1; preserve the full recipe for later recovery")
    p.add_argument("--device", default=None,
                   help="Execution device; defaults to the active accelerator (xpu on PVC)")
    p.add_argument("--dry-run", action="store_true", help="Metadata and angular partitions only")
    p.add_argument("--check-initialization", action="store_true",
                   help="Bounded CPU SDF initialization probe; no radar responses or fitting")
    return p


def main(argv=None):
    """Copy of the original ``main`` with the ``--device`` default changed.

    Kept as a copy rather than a delegation because the only CUDA touch is the
    parser default; every other line, including the exit codes (75 interrupted,
    2 degenerate/incomplete, 0 complete) and the Slurm-allocation requirement,
    is the original's and must stay that way.
    """
    args = build_parser().parse_args(argv)
    acquisition = CollectionAcquisition(object_name=args.object, dataset_root=args.dataset_root,
        npz_path=args.npz_path, manifest=args.parent_role_manifest)
    config = json.loads(args.config.read_text()) if args.config else {}
    recipe = make_recipe(acquisition.kind, config)
    output = args.checkpoint_root or Path("training_checkpoints/RIFT_dataset")/acquisition.contract["dataset_identity"]["object_id"]/"sugavanam_ertin_paper_v1"
    if args.dry_run or args.check_initialization:
        # Bounded probe: CPU model, no radar responses, no device needed.
        _, _, planning = plan(acquisition, recipe)
        planning["output"] = str(output)
        if args.check_initialization:
            model = PaperSDF(**_model_config(recipe, acquisition.extent))
            planning["initialization_audit"] = initialization_audit(model, acquisition.extent, seed=recipe["seed"])
        print(json.dumps(planning, indent=2))
        return 2 if args.check_initialization and planning["initialization_audit"]["status"] == "initialization_degenerate" else 0
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Real SE paper-v1 fitting requires an experiment-manager Slurm allocation")
    from rift.sugavanam_ertin_stage2_runtime_v1 import install_stop_handlers, stop_requested, reset_stop_request
    reset_stop_request()
    install_stop_handlers()
    result = run(acquisition, recipe, output, device=args.device, resume=args.resume,
                 should_stop=stop_requested, stage1_only=args.stage1_only)
    print(json.dumps(result, indent=2))
    return 75 if result["status"] == "interrupted" else (0 if result["status"] in ("complete", "stage1_complete") else 2)
