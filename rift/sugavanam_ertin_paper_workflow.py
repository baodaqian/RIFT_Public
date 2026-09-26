"""New SE sub-aperture -> SDF workflow, exposed by the maintained root CLI.

Real fitting belongs to the experiment manager. Metadata planning and bounded
synthetic validation do not fit real radar responses. Historical SE workflows
remain in their original modules and cannot resume this recipe.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
from collections.abc import Mapping

import numpy as np
import torch

from .sugavanam_ertin_paper import (SCHEMA, PAPER, Subapertures, aggregate_scattering,
    pca_normals, PaperSDF, refresh_surface, sdf_losses, initialization_audit)
from .sugavanam_ertin_sparse import initial_state, constrained_step
from .sugavanam_ertin_acquisition import (CollectionAcquisition, GOTCHAAcquisition,
    digest, array_digest, training_statistics, data_objective, validation_readout)
from .sugavanam_ertin_stage2_runtime_v1 import atomic_torch_save, atomic_json_dump


def make_recipe(kind, config=None):
    """All choices omitted by the paper remain explicit and checkpoint-bound."""
    r = dict(schema=SCHEMA, paper=PAPER, seed=42,
        granularity=40, grid_pitch_rule="one_native_range_resolution",
        azimuth_bins=72, elevation_bins=1,
        stage1_iterations=150, residual_relative_energy=.01, initial_lipschitz=1.,
        residual_rtol=1e-3, optimality_rtol=1e-5,
        validation_every=10, checkpoint_every=10,
        point_chunk=4096, pair_chunk=32,
        threshold_fraction=.15, cloud_max_points=0, normal_radius_m=.3,
        stage2_steps=5000, hidden_dim=512, n_layers=8, n_fourier=9, fourier_scale=2.,
        initialization="standard_gaussian", sdf_lr=1e-4, sdf_adam_eps=1e-8,
        batch_on=2048, batch_off=2048, batch_iso=2048,
        iso_count=2048, iso_every=100, iso_start=1, projection_iterations=24,
        edge_weight="paper_literal", alpha_off=100.,
        lambda_on=1., lambda_normal=1., lambda_off=1., lambda_eik=1.,
        lambda_iso=1., lambda_iso_normal=1., export_grid=48,
        objective="min_sum_complex_modulus_subject_to_residual_energy_budget",
        checkpoint_selection="constrained_terminal_stage1_then_fixed_stage2_budget_validation_diagnostic_only",
        geometry_supervision="radar_cloud_only_no_signed_anchors",
        sdf_loss_coordinates="metric_coordinates_raw_tanh_raw_distance_losses",
        fidelity="synthetic_fixture_not_benchmark" if kind == "synthetic" else "paper_equations_declared_author_gaps",
        stage1_signal="subaperture_diagnostic_only", sdf_signal="not_defined")
    if config is not None and not isinstance(config, Mapping):
        raise ValueError("SE configuration must be a JSON object")
    config = dict(config or {})
    # User-selected training default, disclosed separately from the literal
    # paper initialization. Explicit run configs retain priority; synthetic
    # fixtures keep their historical default.
    if kind != "synthetic":
        config = {"initialization_std": .05, **config}
    if "initialization_std" in config:
        r["initialization_std"] = 1.
    tunable = {k for k, v in r.items() if isinstance(v, (int, float))}
    if kind != "synthetic":
        tunable -= {"hidden_dim", "n_layers", "normal_radius_m", "cloud_max_points"}
        if kind == "gotcha_native":
            tunable -= {"azimuth_bins", "elevation_bins", "n_fourier"}
    if set(config) - tunable:
        raise ValueError(f"Unknown or immutable SE recipe keys: {sorted(set(config)-tunable)}")
    for k, v in config.items():
        if type(r[k]) is int and (type(v) is not int):
            raise ValueError(f"{k} must be an integer")
        if type(r[k]) is float and (type(v) not in (int, float)):
            raise ValueError(f"{k} must be a real number")
        r[k] = v
    for k, v in r.items():
        if isinstance(v, (int, float)):
            if isinstance(v, bool) or not math.isfinite(v) or v < 0:
                raise ValueError(f"{k} must be finite and nonnegative")
            if k not in ("seed", "granularity", "cloud_max_points") and v == 0:
                raise ValueError(f"{k} must be positive")
    if r["granularity"] == 1 or r["export_grid"] < 3 or r["n_layers"] < 4 or r["iso_count"] < 4:
        raise ValueError("Invalid spatial grid/network/iso count")
    if not 0 < r["threshold_fraction"] < 1 or r["cloud_max_points"] in (1, 2):
        raise ValueError("Invalid cloud threshold or cap")
    if not 0 < r["residual_relative_energy"] < 1 or not 0 < r["residual_rtol"] < 1 or not 0 < r["optimality_rtol"] < 1:
        raise ValueError("Invalid residual budget or solver tolerance")
    if not any(r[k] for k in r if k.startswith("lambda_")):
        raise ValueError("At least one SDF loss must be active")
    if r.get("initialization_std", 1.) != 1.:
        r["fidelity"] = "user_requested_gaussian_initialization_std_override"
    elif kind != "synthetic":
        # Explicit std=1 restores the original literal recipe identity, whose
        # records omitted this field. Old checkpoints are not relabeled.
        r.pop("initialization_std", None)
    return r


def grid_points(extent, granularity, device="cpu"):
    pitch = 2*extent/granularity
    axis = torch.arange(granularity, dtype=torch.float64, device=device)*pitch-extent+pitch/2
    return torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)


def plan(acquisition, recipe):
    recipe = dict(recipe)
    if recipe["granularity"] == 0:
        recipe["granularity"] = max(2, math.ceil(2*acquisition.extent/acquisition.range_resolution_m))
    partition = Subapertures.fit(acquisition.directions["train"], recipe["azimuth_bins"], recipe["elevation_bins"],
                                direction_statistics=getattr(acquisition, "direction_statistics", None))
    validation, fallback = partition.assign(acquisition.directions["validation"])
    return partition, validation, dict(schema=SCHEMA, acquisition_identity=digest(acquisition.identity),
        recipe=recipe, subapertures=partition.record(),
        train_views=len(acquisition.keys["train"]), validation_views=len(validation),
        native_view_counts=getattr(acquisition, "native_view_counts", None),
        validation_empty_bin_views=int(fallback.sum()),
        subaperture_train_counts=np.bincount(partition.assignments).tolist(),
        extent_m=acquisition.extent, voxel_pitch_m=2*acquisition.extent/recipe["granularity"],
        test_access=False, response_payload_read=False,
        stage1_output="subaperture complex grids and summed magnitude cloud",
        stage2_output="neural SDF and native zero-level mesh; no complex NVS")


def _torch_load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _finite_tree(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite_tree(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite_tree(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def validate_resume(saved, acquisition, recipe, partition):
    """Validate identity and progress before statistics or response access."""
    if saved.get("schema") != SCHEMA:
        raise ValueError("Not an SE paper-v1 checkpoint; historical recipes cannot resume")
    if saved.get("acquisition") != acquisition.identity or saved.get("recipe") != recipe:
        raise ValueError("SE checkpoint source/object/split/acquisition or recipe changed")
    if acquisition.kind == "rift_collection":
        from .rift_dataset import validate_checkpoint_object
        validate_checkpoint_object({"contract": saved["acquisition"]["contract"]}, acquisition.contract)
    if saved.get("partition") != partition.record():
        raise ValueError("SE checkpoint angular partition changed")
    phase = saved.get("phase")
    if phase not in ("stage1", "stage1_unconverged", "stage2", "complete", "surface_unavailable", "incomplete_iso_supervision"):
        raise ValueError("Invalid SE checkpoint phase")
    groups, voxels = len(partition.directions), recipe["granularity"]**3
    fields = saved.get("fields")
    if not isinstance(fields, torch.Tensor) or fields.shape != (groups, voxels) or fields.dtype != torch.complex128 or not torch.isfinite(fields).all():
        raise ValueError("Invalid SE sub-aperture coefficients")
    epoch, cursor = saved.get("iteration"), saved.get("group_cursor")
    if (type(epoch) is not int or not 0 <= epoch <= recipe["stage1_iterations"]
            or type(cursor) is not int or not 0 <= cursor < groups
            or (epoch == recipe["stage1_iterations"] and cursor != 0)):
        raise ValueError("Invalid SE Stage-1 progress")
    expected = epoch + (partition.assignments < cursor).astype(np.int64)
    if saved.get("view_exposures") != expected.tolist():
        raise ValueError("Stage-1 fitting exposure/cursor mismatch")
    stats = saved.get("statistics", {})
    counts = stats.get("per_view_samples", [])
    if (stats.get("schema") != "se_train_rms_v1" or stats.get("identity") != digest(acquisition.identity)
            or len(counts) != len(expected) or any(type(c) is not int or c <= 0 for c in counts)
            or stats.get("samples") != sum(counts)
            or not isinstance(stats.get("rms"), (float, int)) or not math.isfinite(stats["rms"]) or stats["rms"] <= 0
            or stats.get("kernel_scale") != acquisition.kernel_scale):
        raise ValueError("Invalid SE training normalization")
    if counts != acquisition.train_sample_counts:
        raise ValueError("Normalization measurement counts differ from native acquisition headers")
    history = saved.get("stage1_history", [])
    if len(history) != epoch*groups+cursor or not _finite_tree(history):
        raise ValueError("Stage-1 history does not match completed solver steps")
    for index, row in enumerate(history):
        if row.get("iteration") != index//groups+1 or row.get("group") != index % groups:
            raise ValueError("Stage-1 history order changed")
    solvers = saved.get("sparse_solvers", [])
    if len(solvers) != groups or not _finite_tree(solvers):
        raise ValueError("Invalid sparse-solver recovery state")
    energies = stats.get("per_view_energy", [])
    if len(energies) != len(expected) or any(not math.isfinite(e) or e < 0 for e in energies):
        raise ValueError("Missing training energies for Eq. 4 noise budgets")
    if not math.isclose(sum(energies)/sum(counts), stats["rms"]**2, rel_tol=1e-12):
        raise ValueError("Training RMS differs from the constraint's source energies")
    for group, solver in enumerate(solvers):
        ids = np.flatnonzero(partition.assignments == group)
        target = .5*recipe["residual_relative_energy"]*sum(energies[i] for i in ids)/(sum(counts[i] for i in ids)*stats["rms"]**2)
        if (solver.get("target_loss") != target or solver.get("radius", -1) < 0
                or solver.get("lipschitz", 0) <= 0 or solver.get("lower_radius", -1) < 0
                or type(solver.get("converged")) is not bool or type(solver.get("stalled")) is not bool
                or type(solver.get("root_updates")) is not int or solver["root_updates"] < 0
                or (solver.get("upper_radius") is not None and solver["upper_radius"] < solver["lower_radius"])):
            raise ValueError("Invalid constrained solver budget/state")
        updates = epoch+int(group < cursor)
        if updates:
            row = history[(updates-1)*groups+group]
            if (row.get("sparse_solver") != solver
                    or row.get("fields_sha256") != array_digest(fields[group].numpy())):
                raise ValueError("Constrained solver recovery differs from its last committed step")
        elif solver != initial_state(target, recipe["initial_lipschitz"]) or bool(fields[group].count_nonzero()):
            raise ValueError("Nonzero state in an unvisited sub-aperture")
    best = saved.get("best")
    if best is None and epoch >= min(recipe["validation_every"], recipe["stage1_iterations"]):
            raise ValueError("Missing diagnostic Stage-1 snapshot")
    if best is not None:
        if (not isinstance(best, dict) or best.get("fields", torch.empty(0)).shape != fields.shape
                or best["fields"].dtype != torch.complex128
                or not _finite_tree(best) or type(best.get("iteration")) is not int
                or not 1 <= best["iteration"] <= epoch):
            raise ValueError("Invalid Stage-1 selection checkpoint")
    if phase == "stage1_unconverged" and (epoch != recipe["stage1_iterations"] or all(s["converged"] for s in solvers)):
        raise ValueError("Invalid unconverged sparse-solver terminal state")
    if phase not in ("stage1", "stage1_unconverged"):
        if epoch != recipe["stage1_iterations"] or best is None:
            raise ValueError("Stage 2 requires completed full-view Stage 1 and explicit selection")
        if (not all(s["converged"] for s in solvers) or best["iteration"] != epoch
                or not torch.equal(best["fields"], fields)):
            raise ValueError("Stage 2 requires the terminal constrained solution")
        if not isinstance(saved.get("cloud"), dict) or saved.get("stage1_source") != _stage1_source(best):
            raise ValueError("Stage-1/Stage-2 source linkage changed")
        _validate_cloud(saved["cloud"], acquisition.extent)
        # Recompute the handoff from its selected scattering state, not from a
        # caller-supplied cloud that could have been swapped independently.
        expected_cloud = extract_cloud(best["fields"], partition,
            grid_points(acquisition.extent, recipe["granularity"]), acquisition.extent, recipe)
        from .sugavanam_ertin_stage2_runtime_v1 import same_value
        if not same_value(saved["cloud"], expected_cloud):
            raise ValueError("Stage-2 cloud does not match the selected Stage-1 field")
        step = saved.get("sdf_step")
        if type(step) is not int or not 0 <= step <= recipe["stage2_steps"]:
            raise ValueError("Invalid SDF step")
        model = PaperSDF(**_model_config(recipe, acquisition.extent))
        model.load_state_dict(saved["sdf_model"], strict=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=recipe["sdf_lr"], eps=recipe["sdf_adam_eps"])
        optimizer.load_state_dict(saved["sdf_optimizer"])
        if not _finite_tree(saved["sdf_model"]) or not _finite_tree(saved["sdf_optimizer"]):
            raise ValueError("Nonfinite SDF recovery state")
        groups_state = optimizer.param_groups
        if (len(groups_state) != 1 or groups_state[0]["lr"] != recipe["sdf_lr"]
                or groups_state[0]["eps"] != recipe["sdf_adam_eps"]
                or groups_state[0]["betas"] != (.9, .999) or groups_state[0]["weight_decay"] != 0):
            raise ValueError("SDF optimizer settings changed")
        if (step == 0 and optimizer.state) or (step > 0 and len(optimizer.state) != len(list(model.parameters()))):
            raise ValueError("SDF optimizer recovery is incomplete")
        for parameter, recovery in optimizer.state.items():
            if (float(recovery["step"]) != step or recovery["exp_avg"].shape != parameter.shape
                    or recovery["exp_avg_sq"].shape != parameter.shape or (recovery["exp_avg_sq"] < 0).any()):
                raise ValueError("SDF optimizer moments/step changed")
        history = saved.get("sdf_history", [])
        if len(history) != step or not _finite_tree(history) or any(row.get("step") != i+1 for i, row in enumerate(history)):
            raise ValueError("SDF history does not match the completed steps")
        q, n, valid = (saved.get(k) for k in ("iso_points", "iso_normals", "iso_normal_valid"))
        if q is None:
            if n is not None or valid is not None:
                raise ValueError("Partial iso-point recovery state")
        elif (not all(isinstance(v, torch.Tensor) for v in (q, n, valid))
              or q.ndim != 2 or q.shape[1] != 3 or len(q) < 3 or n.shape != q.shape
              or valid.shape != (len(q),) or valid.dtype != torch.bool
              or not _finite_tree([q, n]) or not (q.abs() < acquisition.extent).all()):
            raise ValueError("Invalid iso-point recovery state")
        generator = torch.Generator()
        generator.set_state(saved["generator_state"])
        if phase in ("complete", "surface_unavailable", "incomplete_iso_supervision") and step != recipe["stage2_steps"]:
            raise ValueError("Terminal SDF checkpoint has an incomplete trajectory")
    return saved


def _stage1_source(best):
    return dict(iteration=best["iteration"], validation=best["validation"],
                fields_sha256=array_digest(best["fields"].cpu().numpy()), selection="constrained_terminal")


def _model_config(recipe, extent):
    return dict(extent=extent, **{k: recipe[k] for k in (
        "n_fourier", "fourier_scale", "hidden_dim", "n_layers", "initialization", "seed")},
        **({"initialization_std": recipe["initialization_std"]}
           if "initialization_std" in recipe else {}))


def _validate_cloud(cloud, extent):
    p, n, m = (cloud.get(k) for k in ("points", "normals", "magnitude"))
    if (not all(isinstance(a, torch.Tensor) for a in (p, n, m)) or p.ndim != 2
            or p.shape[1] != 3 or len(p) < 3 or n.shape != p.shape or m.shape != (len(p),)
            or not _finite_tree(cloud) or not (p.abs() < extent).all() or not (m > 0).all()
            or not torch.allclose(n.norm(dim=-1), torch.ones(len(n)), atol=1e-5, rtol=0)):
        raise ValueError("Invalid radar-derived Stage-1 cloud")


def extract_cloud(fields, partition, points, extent, recipe):
    magnitude, directions, strongest = aggregate_scattering(fields, partition.directions)
    peak = float(magnitude.max())
    if peak <= 0:
        raise ValueError("Stage 1 has no nonzero scattering cloud")
    threshold = recipe["threshold_fraction"]*peak
    selected = torch.nonzero(magnitude >= threshold).flatten()
    before = len(selected)
    if recipe["cloud_max_points"] and before > recipe["cloud_max_points"]:
        order = torch.argsort(magnitude[selected], descending=True, stable=True)
        selected = selected[order[:recipe["cloud_max_points"]]]
    if len(selected) < 3:
        raise ValueError("Stage-1 threshold retained fewer than three scattering points")
    p = points.detach().cpu()[selected]
    radius = recipe["normal_radius_m"]
    n, valid, fallback = pca_normals(p.numpy(), radius, fallback_directions=directions[selected].cpu().numpy())
    result = dict(points=p.float(), normals=torch.from_numpy(n).float(), magnitude=magnitude[selected].cpu(),
        strongest_directions=directions[selected].cpu(), strongest_subaperture=strongest[selected].cpu(),
        threshold=threshold, threshold_count=before, retained_count=len(selected),
        removed_magnitude=float(magnitude[magnitude >= threshold].sum()-magnitude[selected].sum()),
        normal_radius_m=radius, normal_fallback_count=int(fallback.sum()),
        valid_normal_count=int(valid.sum()), aggregation="sum_abs_subapertures")
    _validate_cloud(result, extent)
    return result


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(v) for v in value]
    return copy.deepcopy(value)


def _save(state, root):
    atomic_torch_save(state, root/"checkpoint_latest.pt")
    atomic_json_dump(dict(schema=SCHEMA, phase=state["phase"], iteration=state["iteration"],
        group_cursor=state["group_cursor"], sdf_step=state.get("sdf_step", 0),
        train_views=len(state["view_exposures"]), minimum_fitting_exposures=min(state["view_exposures"]),
        stage1_selection=state.get("stage1_source"), sdf_complex_nvs=False), root/"status.json")


def _surface_export(model, root, extent, resolution, provenance):
    """Native SDF zero readout. Disconnected/open components remain visible."""
    from skimage.measure import marching_cubes
    axis = torch.linspace(-extent, extent, resolution, device=next(model.parameters()).device)
    points = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
    with torch.no_grad():
        values = torch.cat([model(p).cpu() for p in points.split(32768)])
    field = values.numpy().reshape((resolution,)*3)
    finite = bool(np.isfinite(field).all())
    if not finite or not field.min() < 0 < field.max():
        return dict(status="surface_unavailable", finite=finite,
                    reason="sampled_field_has_no_zero_crossing", provenance=provenance)
    spacing = 2*extent/(resolution-1)
    vertices, faces, normals, _ = marching_cubes(field, level=0., spacing=(spacing,)*3)
    vertices -= extent
    # Geometry is reported without a single-component closed-surface policy.
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    status = "complete" if provenance["iso_supervised"] and provenance["iso_normals_supervised"] else "incomplete_iso_supervision"
    audit = dict(status=status, vertices=len(vertices), faces=len(faces),
        boundary_edges=int((counts == 1).sum()), nonmanifold_edges=int((counts > 2).sum()),
        topology_policy="diagnostic_no_single_component_requirement",
        vertex_manifold_and_self_intersections="not_certified", coordinate_units="metres",
        extraction="SDF_zero_marching_cubes", grid=resolution, provenance=provenance)
    temporary = root/"surface.tmp.npz"
    np.savez(temporary, vertices=vertices, faces=faces, normals=normals, sdf_grid=field,
             extent_m=extent, provenance_json=json.dumps(audit, sort_keys=True))
    os.replace(temporary, root/"surface.npz")
    import hashlib
    h = hashlib.sha256()
    with (root/"surface.npz").open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            h.update(block)
    audit["artifact_sha256"] = h.hexdigest()
    return audit


def run(acquisition, recipe, output, *, device="cuda", resume=None, should_stop=None,
        stage1_only=False):
    """Both stages with atomic recovery at complete sub-aperture/optimizer steps.

    ``should_stop`` supports manager TERM boundaries and small synthetic tests;
    it never changes the recipe's final budget. ``stage1_only`` stops at the
    sparse-stage boundary without changing the saved recipe or checkpoint phase.
    No scheduler is called here.
    """
    partition, validation_assignments, planning = plan(acquisition, recipe)
    recipe = planning["recipe"]
    root = Path(output).absolute()
    state = None
    if resume:
        if Path(resume).absolute().parent != root:
            raise ValueError("Resume must use its original output directory")
        state = validate_resume(_torch_load(resume), acquisition, recipe, partition)
        if stage1_only and state["phase"] not in ("stage1", "stage1_unconverged"):
            raise ValueError("Stage-1-only execution requires a Stage-1 checkpoint")
    elif root.exists() and any(root.iterdir()):
        raise FileExistsError("Use a fresh SE paper-v1 output directory or explicit resume")
    # Identity and progress have been checked before any real response read.
    device = torch.device(device)
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
            atomic_json_dump(report, root/"initialization.json")
            atomic_json_dump(report, root/"status.json")
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
    atomic_json_dump(planning, root/"recipe.json")
    if not resume:
        _save(state, root)  # Durable zero-update state before the first inverse solve.
    stop = should_stop or (lambda: False)
    if state["phase"] == "stage1_unconverged":
        return dict(status=state["phase"], output=str(root), resumed_terminal=True)
    if state["phase"] in ("complete", "surface_unavailable", "incomplete_iso_supervision"):
        if state["surface"].get("artifact_sha256"):
            import hashlib
            surface = root/"surface.npz"
            if not surface.is_file() or hashlib.sha256(surface.read_bytes()).hexdigest() != state["surface"]["artifact_sha256"]:
                raise ValueError("Terminal checkpoint surface artifact is missing or changed")
        return dict(status=state["phase"], output=str(root), resumed_terminal=True)
    if state["phase"] == "stage1":
        while state["iteration"] < recipe["stage1_iterations"]:
            group = state["group_cursor"]
            indices = np.flatnonzero(partition.assignments == group).tolist()
            x = state["fields"][group].to(device)
            def value_grad(w):
                return data_objective(acquisition, points, w, indices, state["statistics"], recipe, gradient=True)
            def value(w):
                return data_objective(acquisition, points, w, indices, state["statistics"], recipe)
            x, solver, audit = constrained_step(x, value_grad, value, state["sparse_solvers"][group],
                residual_rtol=recipe["residual_rtol"], optimality_rtol=recipe["optimality_rtol"])
            state["fields"][group] = x.cpu()
            state["sparse_solvers"][group] = solver
            for i in indices:
                state["view_exposures"][i] += 1
            state["stage1_history"].append(dict(iteration=state["iteration"]+1, group=group,
                sparse_solver=dict(solver), fields_sha256=array_digest(x.cpu().numpy()), **audit))
            state["group_cursor"] = (group+1) % groups
            if state["group_cursor"] == 0:
                state["iteration"] += 1
                iteration = state["iteration"]
                if iteration % recipe["validation_every"] == 0 or iteration == recipe["stage1_iterations"]:
                    metrics = validation_readout(acquisition, points, state["fields"], validation_assignments,
                                                 state["statistics"], recipe)
                    state["best"] = dict(iteration=iteration, fields=state["fields"].clone(), validation=metrics)
                    print(json.dumps(dict(stage=1, iteration=iteration, **metrics)), flush=True)
            completed = state["iteration"]*groups + state["group_cursor"]
            stopping = stop()
            if completed % recipe["checkpoint_every"] == 0 or stopping or state["iteration"] == recipe["stage1_iterations"]:
                _save(state, root)
            if stopping:
                return dict(status="interrupted", phase="stage1", output=str(root))
        if not all(s["converged"] for s in state["sparse_solvers"]):
            state["phase"] = "stage1_unconverged"
            _save(state, root)
            atomic_torch_save(state, root/"checkpoint_final.pt")
            return dict(status=state["phase"], output=str(root),
                        reason="Eq4 residual/optimality tolerances not met; no SDF fit or geometry score")
        if stage1_only:
            _save(state, root)
            atomic_torch_save(state, root/"checkpoint_stage1_final.pt")
            return dict(status="stage1_complete", phase="stage1", output=str(root),
                        iteration=state["iteration"], sdf_step=0)
        state["stage1_source"] = _stage1_source(state["best"])
        state["cloud"] = extract_cloud(state["best"]["fields"], partition, points,
                                        acquisition.extent, recipe)
        atomic_torch_save(dict(schema=SCHEMA, acquisition=acquisition.identity, recipe=recipe,
            partition=partition.record(), statistics=state["statistics"],
            stage1_source=state["stage1_source"], selected_fields=state["best"]["fields"],
            terminal_view_exposures=state["view_exposures"], cloud=state["cloud"]),
            root/"stage1_selected.pt")
        model = PaperSDF(**_model_config(recipe, acquisition.extent)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=recipe["sdf_lr"], eps=recipe["sdf_adam_eps"])
        generator = torch.Generator().manual_seed(recipe["seed"])
        state.update(phase="stage2", sdf_step=0, sdf_model=_cpu_tree(model.state_dict()),
            sdf_optimizer=_cpu_tree(optimizer.state_dict()), generator_state=generator.get_state(),
            iso_points=None, iso_normals=None, iso_normal_valid=None)
        # Record the effective numerical initialization; near-zero values alone
        # never count as an initialized surface or a successful projection.
        from .sugavanam_ertin_paper import field_gradient
        probes = ((torch.rand(256, 3, generator=generator)*2-1)*acquisition.extent).to(device)
        f, g = field_gradient(model, probes)
        state["initialization_audit"] = dict(mean_abs_sdf_m=float(f.detach().abs().mean()),
            positive_fraction=float((f.detach() > 0).float().mean()),
            saturated_fraction=float((f.detach().abs() > .99).float().mean()),
            mean_gradient_norm=float(g.detach().norm(dim=-1).mean()),
            initialization=recipe["initialization"])
        state["generator_state"] = generator.get_state()
        _save(state, root)
    else:
        model = PaperSDF(**_model_config(recipe, acquisition.extent)).to(device)
        model.load_state_dict(state["sdf_model"])
        optimizer = torch.optim.Adam(model.parameters(), lr=recipe["sdf_lr"], eps=recipe["sdf_adam_eps"])
        optimizer.load_state_dict(state["sdf_optimizer"])
        generator = torch.Generator()
        generator.set_state(state["generator_state"])
    cloud = state["cloud"]
    on_points, on_normals = cloud["points"].to(device), cloud["normals"].to(device)
    iso = None if state["iso_points"] is None else state["iso_points"].to(device)
    iso_normals = None if state["iso_normals"] is None else state["iso_normals"].to(device)
    iso_valid = None if state["iso_normal_valid"] is None else state["iso_normal_valid"].to(device)
    pitch = 2*acquisition.extent/recipe["granularity"]
    def capture():
        state.update(sdf_model=_cpu_tree(model.state_dict()), sdf_optimizer=_cpu_tree(optimizer.state_dict()),
            generator_state=generator.get_state(), iso_points=_cpu_tree(iso),
            iso_normals=_cpu_tree(iso_normals), iso_normal_valid=_cpu_tree(iso_valid))
        _save(state, root)
    while state["sdf_step"] < recipe["stage2_steps"]:
        step = state["sdf_step"]+1
        model.train()
        if step >= recipe["iso_start"] and (step-recipe["iso_start"]) % recipe["iso_every"] == 0:
            # CPU generator gives identical sampling state independent of GPU count.
            seed_points = on_points.detach().cpu()
            # Resampler uses the same device as the model, but jitter is generated on CPU.
            iso_cpu, audit = refresh_surface_cpu_generator(model, seed_points,
                extent=acquisition.extent, pitch=pitch, recipe=recipe, generator=generator)
            state["iso_history"].append(dict(step=step, **audit))
            if len(iso_cpu) >= 3:
                normals, valid, _ = pca_normals(iso_cpu.numpy(), recipe["normal_radius_m"])
                iso = iso_cpu.to(device)
                iso_normals = torch.from_numpy(normals).float().to(device)
                iso_valid = torch.from_numpy(valid).to(device)
        index = torch.randint(len(on_points), (recipe["batch_on"],), generator=generator).to(device)
        background = ((torch.rand(recipe["batch_off"], 3, generator=generator)*2-1)*acquisition.extent).to(device)
        idx = None if iso is None else torch.randint(len(iso), (recipe["batch_iso"],), generator=generator).to(device)
        losses = sdf_losses(model, on_points[index], on_normals[index], background,
            None if idx is None else iso[idx], None if idx is None else iso_normals[idx],
            extent=acquisition.extent, alpha_off=recipe["alpha_off"],
            iso_normal_valid=None if idx is None else iso_valid[idx])
        total = sum(recipe["lambda_"+k]*v for k, v in losses.items())
        if not torch.isfinite(total):
            raise FloatingPointError("Nonfinite SDF objective; last good checkpoint preserved")
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("Nonfinite SDF gradient; last good checkpoint preserved")
        optimizer.step()
        if not _finite_tree(model.state_dict()) or not _finite_tree(optimizer.state_dict()):
            raise FloatingPointError("Nonfinite SDF optimizer state; last good checkpoint preserved")
        state["sdf_step"] = step
        state["sdf_history"].append(dict(step=step, total=float(total.detach()),
                                       iso_points_in_loss=0 if idx is None else len(idx),
                                       iso_normals_in_loss=0 if idx is None else int(iso_valid[idx].sum()),
                                       **{k: float(v.detach()) for k, v in losses.items()}))
        stopping = stop()
        if step % recipe["checkpoint_every"] == 0 or stopping or step == recipe["stage2_steps"]:
            capture()
        if stopping:
            return dict(status="interrupted", phase="stage2", output=str(root))
    model.eval()
    iso_supervised = any(row["iso_points_in_loss"] > 0 for row in state["sdf_history"])
    iso_normals_supervised = any(row["iso_normals_in_loss"] > 0 for row in state["sdf_history"])
    provenance = dict(schema=SCHEMA, acquisition_sha256=digest(acquisition.identity),
        recipe_sha256=digest(recipe), stage1_source=state["stage1_source"], sdf_step=state["sdf_step"],
        sdf_model_sha256=array_digest(*[v.detach().cpu().numpy() for v in model.state_dict().values()]),
        iso_supervised=iso_supervised, iso_normals_supervised=iso_normals_supervised)
    state["surface"] = _surface_export(model, root, acquisition.extent, recipe["export_grid"], provenance)
    state["phase"] = state["surface"]["status"]
    capture()
    atomic_torch_save(state, root/"checkpoint_final.pt")
    return dict(status=state["phase"], output=str(root), stage1=state["stage1_source"],
                surface=state["surface"], sdf_complex_nvs=False)


def refresh_surface_cpu_generator(model, seeds, *, extent, pitch, recipe, generator):
    """Generate seeds/jitter on CPU; projection moves queries to the SDF device."""
    # Projection casts to the model device. Continue resampling there while all
    # random draws have already happened on CPU at the start of refresh_surface.
    q, audit = refresh_surface(model, seeds, extent=extent, pitch=pitch,
        count=recipe["iso_count"], generator=generator, edge_weight=recipe["edge_weight"],
        projection_iterations=recipe["projection_iterations"])
    return q.cpu(), audit


def main(argv=None):
    from .rift_dataset import DEFAULT_ROOT
    p = argparse.ArgumentParser(description=__doc__)
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
    p.add_argument("--device", default="cuda")
    p.add_argument("--dry-run", action="store_true", help="Metadata and angular partitions only")
    p.add_argument("--check-initialization", action="store_true",
                   help="Bounded CPU SDF initialization probe; no radar responses or fitting")
    args = p.parse_args(argv)
    acquisition = CollectionAcquisition(object_name=args.object, dataset_root=args.dataset_root,
        npz_path=args.npz_path, manifest=args.parent_role_manifest)
    config = json.loads(args.config.read_text()) if args.config else {}
    recipe = make_recipe(acquisition.kind, config)
    output = args.checkpoint_root or Path("training_checkpoints/RIFT_dataset")/acquisition.contract["dataset_identity"]["object_id"]/"sugavanam_ertin_paper_v1"
    if args.dry_run or args.check_initialization:
        _, _, planning = plan(acquisition, recipe)
        planning["output"] = str(output)
        if args.check_initialization:
            model = PaperSDF(**_model_config(recipe, acquisition.extent))
            planning["initialization_audit"] = initialization_audit(model, acquisition.extent, seed=recipe["seed"])
        print(json.dumps(planning, indent=2))
        return 2 if args.check_initialization and planning["initialization_audit"]["status"] == "initialization_degenerate" else 0
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Real SE paper-v1 fitting requires an experiment-manager Slurm allocation")
    from .sugavanam_ertin_stage2_runtime_v1 import install_stop_handlers, stop_requested, reset_stop_request
    reset_stop_request()
    install_stop_handlers()
    result = run(acquisition, recipe, output, device=args.device, resume=args.resume,
                 should_stop=stop_requested, stage1_only=args.stage1_only)
    print(json.dumps(result, indent=2))
    return 75 if result["status"] == "interrupted" else (0 if result["status"] in ("complete", "stage1_complete") else 2)


def run_gotcha(*, dataset, output_dir, config, device, resume):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Real GOTCHA SE fitting requires an experiment-manager Slurm allocation")
    from .sugavanam_ertin_stage2_runtime_v1 import install_stop_handlers, stop_requested, reset_stop_request
    acquisition = GOTCHAAcquisition(dataset)
    recipe = make_recipe(acquisition.kind, config)
    reset_stop_request()
    install_stop_handlers()
    result = run(acquisition, recipe, output_dir, device=device, resume=resume, should_stop=stop_requested)
    # The shared dispatcher otherwise treats every non-interrupted return as a
    # successful process exit. Preserve a failed scientific/numerical status.
    if result["status"] not in ("complete", "interrupted"):
        raise RuntimeError(f"SE comparison incomplete: {result['status']}; see {output_dir}/status.json")
    return result
