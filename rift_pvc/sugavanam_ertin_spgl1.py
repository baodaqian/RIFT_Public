"""SE Eq. 4 solved with the reference SPGL1 package (PVC lane, user-selected 2026-09-22).

The paper states Eq. 4 (minimum complex-modulus L1 norm subject to a residual
energy budget, one field per sub-aperture) but names no solver. The repository
solver (``rift.sugavanam_ertin_sparse.constrained_step``, one projected step
per sub-aperture per iteration) never leaves its first Pareto radius on the
collection scenes and every Stage 1 ends ``stage1_unconverged``
(RIFT_PVC_Adaptation.md sections 19 and 21). On 2026-09-22 the user chose to
fill the unspecified solver with the standard tool: SPGL1 (van den Berg and
Friedlander), Python port https://github.com/drrelyea/spgl1, vendored unchanged
under ``rift_pvc/vendor/spgl1``, called as ``spg_bpdn`` with its defaults.

Only the operator is ours, and it is the declared one: row m of A is
    A_mq = a_m exp(-i k_m (r_m - q.d_m)),
exactly ``rift_pvc.sugavanam_ertin_batched.observation_pairs`` /
``fourier_forward`` (Eq. 2, first-order bistatic extension), stored densely for
one sub-aperture; b = y / rms. The budget is the recipe's:
    0.5 ||A x - b||^2 / N <= target_loss   <=>   ||A x - b|| <= sigma = sqrt(2 N target_loss).

Workflow (independent sub-apertures, so Stage 1 runs as parallel CPU jobs):
``solve_groups`` writes one ``stage1_spgl1/group_XX.pt`` per sub-aperture;
``assemble`` turns all results into a terminal Stage-1 checkpoint in the
ORIGINAL paper-v1 schema (``stage1_iterations`` = 1: one SPGL1 solve per
sub-aperture), so the unchanged convergence gate, cloud extraction and Stage 2
of ``rift.sugavanam_ertin_paper_workflow`` continue from ``checkpoint_latest.pt``.
A sub-aperture counts as converged when SPGL1 exits with a root or a BP
solution; the repository's residual/Frank--Wolfe certificates are recorded
alongside as diagnostics, not gates.
"""
from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator

from rift.sugavanam_ertin_acquisition import array_digest, digest
from rift.sugavanam_ertin_paper import Subapertures
from rift.sugavanam_ertin_paper_workflow import SCHEMA, grid_points, plan, validate_resume
from rift.sugavanam_ertin_paper_workflow import make_recipe as paper_recipe
from rift.sugavanam_ertin_stage2_runtime_v1 import atomic_json_dump, atomic_torch_save
from rift_pvc.sugavanam_ertin_batched import _collect, batched_data_objective, batched_validation_readout

SOLVER = 'spgl1_spg_bpdn_defaults'
SPGL1_REPOSITORY = 'https://github.com/drrelyea/spgl1'
SPGL1_COMMIT = '405ca805a2d56d783a0445e834c801c5b7c2263a'
# spgl1.spgl1 exit codes: 1 root found, 2 BP solution found (both satisfy ||r|| <= sigma
# to SPGL1's opt_tol); every other exit (iterations, line search, least squares,
# suboptimal BP, matvec limit, active set) is reported as not converged.
EXIT_NAMES = {1: 'root_found', 2: 'bp_solution_found', 3: 'least_squares', 4: 'optimal',
              5: 'iteration_limit', 6: 'line_search_error', 7: 'suboptimal_bp', 8: 'matvec_limit',
              9: 'active_set'}
ACCEPTED_EXITS = (1, 2)
GROUPS_DIR = 'stage1_spgl1'


def make_recipe(kind, config=None):
    """The paper-v1 recipe with the Stage-1 solver made explicit (a distinct identity)."""
    r = paper_recipe(kind, config)
    r.update(stage1_iterations=1, stage1_solver=SOLVER, spgl1_repository=SPGL1_REPOSITORY,
             spgl1_commit=SPGL1_COMMIT, spgl1_settings='package_defaults_complex_variables',
             spgl1_accepted_exits=[EXIT_NAMES[c] for c in ACCEPTED_EXITS],
             stage1_gate='spgl1_exit_status_repository_certificates_diagnostic')
    return r


def dense_operator(acquisition, points, indices, statistics):
    """(A [M, N] complex128, b [M]) of one sub-aperture, in the batched objective's row order."""
    observations = (o for i in indices for o in acquisition.observations(acquisition.keys["train"][i], role="train"))
    groups = _collect(acquisition, observations, points.device)
    rows = sum(g['t'].numel() for g in groups)
    A = torch.empty((rows, len(points)), dtype=torch.complex128, device=points.device)
    b = torch.empty(rows, dtype=torch.complex128, device=points.device)
    start = 0
    for g in groups:
        k = 2*math.pi*torch.as_tensor(g['f'], device=points.device, dtype=torch.float64)/g['cc']
        for p in range(len(g['r'])):
            path = g['r'][p]-points @ g['d'][p]
            block = torch.exp(-1j*k[:, None]*path[None])*g['a'][p]
            # Row (f, p) of the [F, P] prediction; frequency-major like target.reshape(-1).
            A[start+p:start+len(k)*len(g['r']):len(g['r'])] = block
        b[start:start+g['t'].numel()] = (g['t']/statistics["rms"]).reshape(-1)
        start += g['t'].numel()
    return A, b


def linear_operator(A):
    """scipy view of a CPU torch matrix through numpy BLAS (shared memory).

    torch's CPU complex128 matrix-vector product is ~10x slower than zgemv
    (measured 2026-09-22); A^H r is conj(conj(r)^T A), so A^H is never formed.
    """
    An = A.numpy()
    def matvec(v):
        return An @ np.asarray(v, dtype=np.complex128).reshape(-1)
    def rmatvec(v):
        return (np.asarray(v, dtype=np.complex128).reshape(-1).conj() @ An).conj()
    return LinearOperator(A.shape, matvec=matvec, rmatvec=rmatvec, dtype=np.complex128)


def sigma_from_target(target_loss, samples):
    return math.sqrt(2*samples*target_loss)


def solve(A, b, sigma, **log):
    """spgl1.spg_bpdn with the package defaults (complex variables declared)."""
    from rift_pvc.vendor.spgl1 import spg_bpdn
    x, r, g, info = spg_bpdn(linear_operator(A), b.numpy(), sigma, iscomplex=True, **log)
    return x, info


def repository_diagnostics(A, b, x, radius, target_loss, samples):
    """The repository's Eq. 4 certificates at the SPGL1 solution, as diagnostics only."""
    An = A.numpy()
    xn = np.asarray(x, dtype=np.complex128)
    residual = An @ xn-b.numpy()
    f = .5*float(np.vdot(residual, residual).real)/samples
    gradient = (residual.conj() @ An).conj()/samples
    gap = max(0., float(np.vdot(xn, gradient).real)+radius*float(np.abs(gradient).max()))
    return dict(data_loss=f, residual_relative_error=(f-target_loss)/target_loss,
                frank_wolfe_gap=gap, frank_wolfe_gap_relative=gap/max(f, target_loss),
                l1_norm=float(np.abs(xn).sum()), nonzeros=int((xn != 0).sum()))


def operator_check(acquisition, points, ids, statistics, recipe, A, b, group):
    """Dense operator against the batched objective (value and gradient) at a seeded random field."""
    generator = torch.Generator().manual_seed(group)
    x = (torch.randn(len(points), generator=generator, dtype=torch.float64)
         + 1j*torch.randn(len(points), generator=generator, dtype=torch.float64))*1e-3
    samples = A.shape[0]
    reference, reference_grad = batched_data_objective(acquisition, points, x, ids, statistics, recipe, gradient=True)
    An, xn = A.numpy(), x.numpy()
    residual = An @ xn-b.numpy()
    value = .5*float(np.vdot(residual, residual).real)/samples
    grad = torch.from_numpy((residual.conj() @ An).conj()/samples)
    check = dict(value_relative_difference=abs(value-reference)/abs(reference),
                 gradient_relative_difference=float((grad-reference_grad).norm()/reference_grad.norm()))
    if not (check["value_relative_difference"] <= 1e-9 and check["gradient_relative_difference"] <= 1e-8):
        raise ValueError(f"Dense operator of sub-aperture {group} differs from the batched objective: {check}")
    return check


def group_target(statistics, recipe, ids):
    """The recipe's Eq. 4 budget, with the workflow's exact expression (resume checks equality)."""
    return .5*recipe["residual_relative_energy"]*sum(statistics["per_view_energy"][i] for i in ids)/(
        sum(statistics["per_view_samples"][i] for i in ids)*statistics["rms"]**2)


def exclusive_save(payload, path):
    """Write ``path`` once: the first writer wins, later writers keep what is on disk.

    The payload goes to a temporary file that is hard-linked into place; the link
    fails if ``path`` already exists, so no reader ever sees a partial or
    overwritten file even when parallel jobs start together.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(dict(payload), temporary)
        os.link(temporary, path)
    except FileExistsError:
        pass
    finally:
        os.unlink(temporary)


PARTITION_DIRECTION_ATOL = 1e-12


def partition_difference(pinned, here):
    """How a recomputed partition record differs from the pinned one."""
    discrete = [k for k in pinned if k != "directions" and pinned[k] != here.get(k)]
    a, b = np.asarray(pinned["directions"]), np.asarray(here["directions"])
    gap = float(np.abs(a-b).max()) if a.shape == b.shape else math.inf
    return dict(discrete_fields_differing=discrete, directions_max_abs_difference=gap,
                directions_rows_differing=int((a != b).any(1).sum()) if a.shape == b.shape else None)


def pinned_partition(acquisition, recipe, output):
    """The angular partition fixed by the first job that computes it (``stage1_spgl1/partition.pt``).

    ``Subapertures.fit``'s mean directions are not bitwise reproducible across nodes:
    numpy/libm pick CPU-specific SIMD kernels, and the last bits change (measured
    2026-09-22: A320 gave two record digests on two cpu nodes, and a third with
    numpy's AVX-512 dispatch disabled; the sub-aperture assignments were identical
    each time). Every solve and assemble therefore binds the pinned record. A
    recomputation must reproduce its bins and assignments exactly and its directions
    within ``PARTITION_DIRECTION_ATOL``; otherwise this raises.
    Returns (partition built from the pinned record, validation assignments, planning).
    """
    partition, _, planning = plan(acquisition, recipe)
    path = Path(output)/GROUPS_DIR/"partition.pt"
    if not path.exists():
        exclusive_save(dict(acquisition=digest(acquisition.identity), record=partition.record()), path)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved["acquisition"] != digest(acquisition.identity):
        raise ValueError(f"{path} belongs to another acquisition")
    pinned = saved["record"]
    difference = partition_difference(pinned, partition.record())
    if difference["discrete_fields_differing"] or not difference["directions_max_abs_difference"] <= PARTITION_DIRECTION_ATOL:
        raise ValueError(f"Recomputed sub-aperture partition differs from the pinned one in {path}: {difference}")
    fixed = Subapertures(pinned["azimuth_bins"], pinned["elevation_bins"], np.asarray(pinned["occupied_bins"]),
                         np.asarray(pinned["directions"]), np.asarray(pinned["train_assignments"]))
    validation_assignments, _ = fixed.assign(acquisition.directions["validation"])
    return fixed, validation_assignments, dict(planning, subapertures=fixed.record())


def identity(acquisition, recipe, partition, statistics):
    return dict(acquisition=digest(acquisition.identity), recipe=digest(recipe),
                partition=digest(partition.record()), statistics=digest(statistics))


def solve_groups(acquisition, recipe, statistics, output, groups, *, verbosity=1, log=print):
    """Solve and persist the requested sub-apertures; completed results are kept, not redone.

    Each dense operator is first checked against the batched objective; a mismatch stops the job.
    """
    partition, _, planning = pinned_partition(acquisition, recipe, output)
    recipe = planning["recipe"]
    ident = identity(acquisition, recipe, partition, statistics)
    folder = Path(output)/GROUPS_DIR
    folder.mkdir(parents=True, exist_ok=True)
    points = grid_points(acquisition.extent, recipe["granularity"], "cpu")
    for group in groups:
        path = folder/f"group_{group:02d}.pt"
        if path.exists():
            if torch.load(path, map_location="cpu", weights_only=False)["identity"] != ident:
                raise ValueError(f"{path} belongs to another acquisition/recipe/partition")
            log(dict(event="group_kept", group=group))
            continue
        ids = np.flatnonzero(partition.assignments == group).tolist()
        samples = sum(statistics["per_view_samples"][i] for i in ids)
        target = group_target(statistics, recipe, ids)
        A, b = dense_operator(acquisition, points, ids, statistics)
        if A.shape[0] != samples:
            raise ValueError("Operator rows differ from the group's native sample count")
        check = operator_check(acquisition, points, ids, statistics, recipe, A, b, group)
        sigma = sigma_from_target(target, samples)
        x, info = solve(A, b, sigma, verbosity=verbosity)
        stat = int(info["stat"])
        spgl1 = dict(stat=stat, exit=EXIT_NAMES.get(stat, str(stat)), niters=int(info["niters"]),
                     nprodA=int(info["nprodA"]), nprodAt=int(info["nprodAt"]), n_newton=int(info["n_newton"]),
                     tau=float(info["tau"]), rnorm=float(info["rnorm"]), rgap=float(info["rgap"]),
                     sigma=sigma, time_total=float(info["time_total"]), time_matprod=float(info["time_matprod"]))
        diagnostics = repository_diagnostics(A, b, x, spgl1["tau"], target, samples)
        atomic_torch_save(dict(identity=ident, group=group, views=len(ids), samples=samples, target_loss=target,
                               field=torch.from_numpy(np.asarray(x, dtype=np.complex128)), spgl1=spgl1,
                               diagnostics=diagnostics, operator_check=check, solver=SOLVER,
                               spgl1_commit=SPGL1_COMMIT), path)
        log(dict(event="group_done", group=group, **spgl1, diagnostics=diagnostics, operator_check=check))
        del A, b


def assemble(acquisition, recipe, statistics, output, *, device="cpu"):
    """Terminal Stage-1 checkpoint (original schema) from all per-sub-aperture SPGL1 results."""
    partition, validation_assignments, planning = pinned_partition(acquisition, recipe, output)
    recipe = planning["recipe"]
    ident = identity(acquisition, recipe, partition, statistics)
    root = Path(output).absolute()
    groups = len(partition.directions)
    results = []
    for group in range(groups):
        path = root/GROUPS_DIR/f"group_{group:02d}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Sub-aperture {group} has no SPGL1 result ({path})")
        result = torch.load(path, map_location="cpu", weights_only=False)
        if result["identity"] != ident or result["group"] != group:
            raise ValueError(f"{path} belongs to another acquisition/recipe/partition")
        results.append(result)
    fields = torch.stack([r["field"] for r in results])
    solvers, history = [], []
    for group, result in enumerate(results):
        ids = np.flatnonzero(partition.assignments == group).tolist()
        target = group_target(statistics, recipe, ids)
        spgl1, diag = result["spgl1"], result["diagnostics"]
        converged = spgl1["stat"] in ACCEPTED_EXITS
        # Keys the original resume validation reads; lipschitz is the recipe's
        # unused initial value (SPGL1 keeps its own spectral step), radius is SPGL1's tau.
        solver = dict(target_loss=target, radius=spgl1["tau"], lipschitz=float(recipe["initial_lipschitz"]),
                      lower_radius=0., upper_radius=None, root_updates=spgl1["n_newton"],
                      converged=converged, stalled=False, solver=SOLVER, spgl1_exit=spgl1["exit"],
                      spgl1_niters=spgl1["niters"], spgl1_rnorm=spgl1["rnorm"], spgl1_sigma=spgl1["sigma"],
                      spgl1_rgap=spgl1["rgap"])
        solvers.append(solver)
        history.append(dict(iteration=1, group=group, sparse_solver=solver,
            fields_sha256=array_digest(fields[group].numpy()), execution=SOLVER,
            data_loss=diag["data_loss"], target_data_loss=target,
            residual_relative_error=diag["residual_relative_error"],
            residual_feasible=diag["data_loss"] <= target*(1+recipe["residual_rtol"]),
            l1_norm=diag["l1_norm"], evaluated_l1_radius=spgl1["tau"],
            subproblem_duality_gap=diag["frank_wolfe_gap"],
            subproblem_stationary=diag["frank_wolfe_gap"] <= recipe["optimality_rtol"]*max(diag["data_loss"], target),
            converged=converged, stalled=False, backtracks=0, root_updates=spgl1["n_newton"],
            spgl1=spgl1))
    points = grid_points(acquisition.extent, recipe["granularity"], device)
    metrics = batched_validation_readout(acquisition, points, fields, validation_assignments, statistics, recipe)
    state = dict(schema=SCHEMA, acquisition=acquisition.identity, recipe=recipe, partition=partition.record(),
                 statistics=statistics, phase="stage1", fields=fields, sparse_solvers=solvers,
                 iteration=recipe["stage1_iterations"], group_cursor=0,
                 view_exposures=[recipe["stage1_iterations"]]*len(acquisition.keys["train"]),
                 best=dict(iteration=recipe["stage1_iterations"], fields=fields.clone(), validation=metrics),
                 stage1_history=history, sdf_history=[], iso_history=[])
    validate_resume(state, acquisition, recipe, partition)  # The original workflow must accept it.
    root.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(planning, root/"recipe.json")
    atomic_json_dump(dict(schema=SCHEMA, execution=SOLVER, spgl1_repository=SPGL1_REPOSITORY,
        spgl1_commit=SPGL1_COMMIT, operator="dense_eq2_subaperture_numpy_blas_cpu",
        stage1="one_spgl1_spg_bpdn_solve_per_subaperture_package_defaults",
        later_stages="original_workflow_resumed_from_checkpoint_latest"), root/"execution.json")
    from rift.sugavanam_ertin_paper_workflow import _save
    _save(state, root)
    return dict(status="stage1_assembled", converged=sum(s["converged"] for s in solvers), groups=groups,
                exits={name: sum(s["spgl1_exit"] == name for s in solvers) for name in EXIT_NAMES.values()},
                validation=metrics, output=str(root))
