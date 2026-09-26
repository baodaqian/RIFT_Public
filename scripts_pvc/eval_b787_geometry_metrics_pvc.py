#!/usr/bin/env python3
"""PVC twin of scripts/eval_b787_geometry_metrics.py that also scores Radar Fields grids.

The original (unchanged on disk) loads every checkpoint through
scripts/render_b787_vs_stl.load_energy_field, which checks a collection
checkpoint's training selection in RIFT's sealed_npz_protocol_contract schema.
Radar Fields records a different schema under sealed_protocol_contract, so a
train2400 Radar Fields checkpoint stops with "missing fields" before any metric.
This twin rebinds one function, a copy of load_energy_field that checks a
Radar Fields checkpoint in its own terms (the requested role-manifest name, which
fixes object, seed, role counts and antenna digest, and the antenna selection;
the object identity is still checked by validate_checkpoint_object), and then
runs the original main unchanged (same CLI, readouts and metrics).

Two more field sources reach the same thresholded-field protocol through
load_any_field (the copied load_energy_field is unchanged for every other
checkpoint): a SpINR checkpoint (its signed field evaluated on its own 48^3
midpoint grid, which is the evaluator's lattice; energy = sigma^2, so the evaluator's
sqrt(energy) is |sigma|) and a RadarSplat geometry_g48.npz export (its Gaussian
occupancy-union support on the same lattice; energy = support^2). Pass either path
to --checkpoints.

    python scripts_pvc/eval_b787_geometry_metrics_pvc.py --object a320 --dataset-root D \
        --num-train 2400 --num-tx 1 --num-rx 1 --checkpoints CKPT [...] --fixed-threshold 0.2 \
        --point-grid 48 --csv OUT.csv
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import scripts.eval_b787_geometry_metrics as original  # noqa: E402
from scripts.render_b787_vs_stl import _point_sh_energy_field  # noqa: E402


def validate_radar_fields_selection(saved, expected_contract):
    """A Radar Fields checkpoint must come from the requested object's manifest and antennas."""
    expected_name = expected_contract.get("role_manifest_name")
    if not expected_name or saved.get("manifest_name") != expected_name:
        raise ValueError(f"Radar Fields checkpoint manifest {saved.get('manifest_name')!r} is not the "
                         f"requested {expected_name!r}")
    expected_antennas = expected_contract.get("antenna_selection")
    saved_antennas = (saved.get("acquisition_identity") or {}).get("antenna_selection")
    if expected_antennas is not None and saved_antennas != expected_antennas:
        raise ValueError("Radar Fields checkpoint antenna selection disagrees with the requested acquisition")


def load_energy_field(checkpoint_path, *, expected_contract=None, extent=None,
                      point_grid=48, readout_info=None):
    """Grid or point-SH checkpoint -> [G,G,G] rotation-invariant energy.

    Grid fields retain their native lattice. Point-SH uses a declared CIC grid,
    not a reshape of slots/anchors and not an inference from occupied bounds.
    The optional info dict records the extraction without changing the old tuple API.
    """
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if expected_contract is not None:
        from rift.rift_dataset import validate_checkpoint_object
        from scripts.eval_b787_range_power import resolve_scene_extent
        validate_checkpoint_object(ck, expected_contract)
        if "sealed_npz_protocol_contract" not in ck and isinstance(ck.get("sealed_protocol_contract"), dict):
            # PVC adaptation: Radar Fields keeps its own contract schema
            # (train_radar_fields.sealed_protocol_contract); check its selection in its own terms.
            validate_radar_fields_selection(ck["sealed_protocol_contract"], expected_contract)
        else:
            saved_contract = ck.get("sealed_npz_protocol_contract", {})
            if "training_selection" in expected_contract or "training_selection" in saved_contract:
                from train import _validate_saved_sealed_npz_protocol_contract
                _validate_saved_sealed_npz_protocol_contract(saved_contract, expected_contract)
        if extent is not None and not np.isclose(resolve_scene_extent(ck), extent, rtol=0, atol=1e-12):
            raise ValueError("Geometry extent disagrees with the selected checkpoint")
    sd = ck["model_state_dict"]
    if ck.get("scene_repr") == "point_sh" or "anchors" in sd:
        from scripts.eval_b787_range_power import resolve_scene_extent
        if ck.get("scene_repr") not in (None, "point_sh"):
            raise ValueError("Checkpoint representation disagrees with point-SH state")
        saved_extent = resolve_scene_extent(ck)
        if extent is not None and not np.isclose(saved_extent, extent, rtol=0, atol=1e-12):
            raise ValueError("Geometry extent disagrees with the selected checkpoint")
        volume, info = _point_sh_energy_field(sd, saved_extent, point_grid)
        if readout_info is not None:
            readout_info.update(info)
        return volume, int(point_grid), ck.get("epoch")
    w_re, w_im = sd["w_re"], sd["w_im"]  # [G,G,G,n_basis]
    if w_re.shape != w_im.shape or w_re.ndim not in (3, 4) or len(set(w_re.shape[:3])) != 1:
        raise ValueError("This geometry readout requires grid, grid_sh, or point_sh state")
    energy = w_re ** 2 + w_im ** 2
    if energy.ndim == 4:
        energy = energy.sum(dim=-1)  # Parseval: direction-integrated power
    if expected_contract is not None and "active_mask" in sd:
        energy = energy * sd["active_mask"].reshape(energy.shape)
    g = w_re.shape[0]  # [G,G,G,n_basis]
    if readout_info is not None:
        readout_info.update(scene_repr=ck.get("scene_repr") or ("grid_sh" if w_re.ndim == 4 else "grid"),
                            spatial_readout="native_voxel_centers", readout_grid=int(g))
    return energy.numpy(), int(g), ck.get("epoch")


def _spinr_field(ck, expected_contract, extent, readout_info):
    import train_spinr_style_pvc as spinr
    from rift.rift_dataset import collection_contract, validate_checkpoint_object
    if expected_contract is not None:
        validate_checkpoint_object(ck, expected_contract)
        if collection_contract(ck.get("sealed_npz_protocol_contract", {})) != collection_contract(expected_contract):
            raise ValueError("SpINR checkpoint was not trained on the requested object/split")
    operator = ck["spinr_style_recipe"]["operator"]
    support = float(ck["spinr_style_recipe"]["network"]["support_m"])
    if operator["quadrature"] != "midpoint" or (extent is not None and not np.isclose(support, extent, rtol=0, atol=1e-12)):
        raise ValueError("SpINR geometry readout needs its midpoint grid on the evaluator extent")
    grid = int(operator["grid_size"])
    points, _ = spinr.midpoint_grid(grid, support_m=support, device="cpu", dtype=torch.float64)
    model = spinr.SpinrStyleINR().to(dtype=torch.float32)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    with torch.no_grad():
        field = spinr.evaluate_neural_field_tiled(model, points, neural_point_tile=8192).double()
    energy = field.square().reshape(grid, grid, grid).numpy()   # cartesian_prod(x, y, z): the evaluator's ij lattice
    if readout_info is not None:
        readout_info.update(scene_repr="spinr_signed_field", spatial_readout="spinr_midpoint_grid_native",
                            readout_grid=grid, energy="sigma^2")
    return energy, grid, int(ck["epoch_index"])


def _radarsplat_field(path, expected_contract, extent, readout_info):
    import json
    from rift.rift_dataset import validate_checkpoint_object
    with np.load(path, allow_pickle=False) as saved:
        identity = json.loads(str(saved["identity_json"]))
        support = saved["support"].astype(np.float64)
        centers = saved["sample_centers_m"].astype(np.float64)
        saved_extent, step = float(saved["extent_m"]), int(saved["step"])
    if expected_contract is not None:
        validate_checkpoint_object({"sealed_protocol_identity": identity["dataset_identity"]}, expected_contract)
    grid = support.shape[0]
    lattice = -saved_extent + (np.arange(grid) + .5) * (2 * saved_extent / grid)
    if (support.shape != (grid,) * 3 or not np.allclose(centers, lattice, rtol=0, atol=1e-12)
            or (extent is not None and not np.isclose(saved_extent, extent, rtol=0, atol=1e-12))):
        raise ValueError("RadarSplat support is not on the evaluator lattice")
    if readout_info is not None:
        readout_info.update(scene_repr="radarsplat_occupancy_union_support", spatial_readout="native_voxel_centers",
                            readout_grid=grid, energy="support^2")
    return np.square(support), grid, step


def load_any_field(checkpoint_path, *, expected_contract=None, extent=None, point_grid=48, readout_info=None):
    """SpINR checkpoints and RadarSplat geometry exports, else the copied load_energy_field."""
    path = str(checkpoint_path)
    if path.endswith(".npz"):
        with np.load(path, allow_pickle=False) as saved:
            schema = str(saved["schema"]) if "schema" in saved.files else None
        if schema != "rift_radarsplat_released_geometry_v1":
            raise ValueError(f"unsupported geometry export {schema!r}")
        return _radarsplat_field(path, expected_contract, extent, readout_info)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("format") == "rift_spinr_style_b787_v2":
        return _spinr_field(ck, expected_contract, extent, readout_info)
    del ck
    return load_energy_field(checkpoint_path, expected_contract=expected_contract, extent=extent,
                             point_grid=point_grid, readout_info=readout_info)


def main(argv=None):
    original.load_energy_field = load_any_field
    original.main(argv)


if __name__ == "__main__":
    main()
