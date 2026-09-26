#!/usr/bin/env python3
"""Measure Radar Fields in its native range-power intensity domain.

This is a post-training readout.  It does not participate in training and does
not change the released objective.  It reports whole-ROI, valid-region, and
padded-region errors, with a zero-RCS reference for every region.
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rift.encoding import generate_dynamic_grid
from rift.radar_fields import (
    NORMALIZED_DB_INTENSITY_DOMAIN,
    NORMALIZED_DB_INTENSITY_LABEL,
    NORMALIZED_DB_RELMSE_LABEL,
    RadarFieldsModel,
    bistatic_range_cells,
    radar_fields_intensity,
)
from rift.radar_fields_dataset import (
    dataset_provenance,
    load_radar_fields_npz,
    load_radar_fields_sealed_split_manifest,
    normalize_power_db,
    range_bin_centers,
    range_bin_size,
    restrict_radar_fields_response_views,
    response_view_to_range_power,
    scene_range_mask,
)
from rift.radar_fields_recipe import AUDITED_RECIPE, native_recipe, AUDITED_FIELDS, recipe_name, recipe_contract, validate_recipe_checkpoint


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "protocols" / "radar_fields_b7873200_production_v1.json"

CONFIG_ARG_KEYS = (
    "sealed_protocol",
    "seed",
    "num_train",
    "num_val",
    "num_test",
    "val_from_tail",
    "extent",
    "granularity",
    "steps",
    "view_batch",
    "train_pairs",
    "val_pairs",
    "eval_every",
    "checkpoint_every",
    "eval_max_views",
    "stats_max_views",
    "lr",
    "weight_fft",
    "weight_occ",
    "weight_bimodal",
    "occupancy_threshold",
    "dynamic_range_db",
    "range_margin",
    "range_law",
    "intensity_offset",
    "intensity_scaler",
    "hidden_dim",
    "feature_dim",
    "sh_degree",
    "sigmoid_tightness",
    "no_batch_norm",
    "hash_levels",
    "hash_features",
    "hash_base_resolution",
    "hash_final_resolution",
    "hash_log2_size",
    "query_chunk",
    "pair_chunk",
    "device",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    require(isinstance(value, Mapping), f"{path} must contain a JSON object")
    return value


def compatible(left: object, right: object) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1.0e-10, abs_tol=1.0e-12)
    return left == right


def _same_resolved_path(saved: object, current: str) -> bool:
    """Path spelling (symlinks, relative vs. absolute) is not identity."""

    if not isinstance(saved, (str, Path)):
        return False
    return Path(saved).resolve() == Path(current).resolve()


def validate_checkpoint(
    checkpoint: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    npz_path: str,
    manifest_path: str,
) -> None:
    training = config["training"]
    data = config["data"]
    require(checkpoint.get("scene_repr") == "radar_fields", "checkpoint is not a Radar Fields checkpoint")
    if native_recipe(SimpleNamespace(**training)):
        require(0 < int(checkpoint.get("step", -1)) <= int(training["steps"]),
                "checkpoint step is outside the configured training schedule")
    else:
        require(int(checkpoint.get("step", -1)) == int(training["steps"]), "checkpoint is not at the proposed final step")
    saved_args = checkpoint.get("args")
    require(isinstance(saved_args, Mapping), "checkpoint lacks persisted trainer arguments")
    validate_recipe_checkpoint(checkpoint, SimpleNamespace(**training))
    if native_recipe(SimpleNamespace(**training)):
        for key in ("recipe", *AUDITED_FIELDS):
            require(key in saved_args and key in training and compatible(saved_args[key], training[key]),
                    f"checkpoint audited configuration mismatch: {key}")
    for key in CONFIG_ARG_KEYS:
        require(key in saved_args, f"checkpoint lacks persisted argument {key}")
        require(compatible(saved_args[key], training[key]), f"checkpoint argument {key} disagrees with production config")
    require(saved_args.get("checkpoint_name") == training["checkpoint_name"], "checkpoint name disagrees with config")
    require(_same_resolved_path(saved_args.get("npz_path"), npz_path), "checkpoint dataset path disagrees with readout input")
    require(_same_resolved_path(saved_args.get("sealed_split_manifest"), manifest_path), "checkpoint manifest path disagrees with readout input")

    split = checkpoint.get("split_provenance")
    require(isinstance(split, Mapping), "checkpoint lacks split provenance")
    roles = split.get("role_ids")
    require(isinstance(roles, Mapping), "checkpoint split provenance lacks role IDs")
    require({name: len(roles[name]) for name in ("train", "val", "test")} == {"train": training["num_train"], "val": training["num_val"], "test": training["num_test"]}, "checkpoint role counts disagree with config")
    require(split.get("test_payload_materialized") is False, "training checkpoint materialized the sealed test payload")

    stats = checkpoint.get("power_stats")
    require(isinstance(stats, Mapping), "checkpoint lacks power statistics")
    require(math.isfinite(float(stats["peak_power"])) and float(stats["peak_power"]) > 0.0, "invalid normalization peak")
    require(list(stats["train_view_indices"]) == list(roles["train"]), "normalization train IDs disagree with split")
    require(list(stats["normalization_scan_view_indices"]) == list(roles["train"]), "normalization was not over all train IDs")

    require(isinstance(checkpoint.get("radar_fields_state_dict"), Mapping), "checkpoint lacks trainable Radar Fields state")
    require(isinstance(checkpoint.get("sealed_protocol_contract"), Mapping), "checkpoint lacks sealed protocol contract")
    require(checkpoint["sealed_protocol_contract"].get("test_response_materialized") is False, "sealed contract exposes test responses")


def metric_accumulator() -> MutableMapping[str, float]:
    return {"abs_error": 0.0, "sq_error": 0.0, "target_power": 0.0, "count": 0.0}


def add_metrics(
    accumulator: MutableMapping[str, float],
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    selected_prediction = prediction[mask].detach().float()
    selected_target = target[mask].detach().float()
    if selected_prediction.numel() == 0:
        return
    error = selected_prediction - selected_target
    accumulator["abs_error"] += float(error.abs().sum().item())
    accumulator["sq_error"] += float(error.square().sum().item())
    accumulator["target_power"] += float(selected_target.square().sum().item())
    accumulator["count"] += float(selected_target.numel())


def finish_metrics(accumulator: Mapping[str, float]) -> Dict[str, float] | None:
    count = int(accumulator["count"])
    if count == 0:
        return None
    sq_error = float(accumulator["sq_error"])
    target_power = float(accumulator["target_power"])
    mse = sq_error / count
    return {
        "measurement_count": count,
        "mean_abs_error": float(accumulator["abs_error"] / count),
        "rel_mse": float(sq_error / max(target_power, 1.0e-30)),
        "rmse": float(math.sqrt(mse)),
        "psnr_db": float(-10.0 * math.log10(max(mse, 1.0e-30))),
        "sq_error": sq_error,
        "target_power": target_power,
    }


def evaluate_view(
    model: RadarFieldsModel,
    arrays,
    view_index: int,
    *,
    xyz: torch.Tensor,
    ranges: torch.Tensor,
    stats: Mapping[str, Any],
    args: SimpleNamespace,
    device: torch.device,
) -> Dict[str, Any]:
    pair_indices = np.arange(arrays.num_tx * arrays.num_rx, dtype=np.int64)
    if native_recipe(args):
        from train_radar_fields import audited_view_tensors
        record = audited_view_tensors(model, arrays, view_index, pair_indices, ranges, stats,
                                      args, 1.0, device)
        valid = record["valid"]
        result = {"view_id": int(view_index), "roi_range_bin_count": int(record["roi"].sum()), "regions": {}}
        zero = torch.full_like(record["prediction"], math.log10(args.intensity_offset) * args.intensity_scaler)
        for name, mask in {"whole_roi": torch.ones_like(valid), "valid_region": valid, "padded_region": ~valid}.items():
            predicted, reference = metric_accumulator(), metric_accumulator()
            add_metrics(predicted, record["prediction"], record["target"], mask)
            add_metrics(reference, zero, record["target"], mask)
            result["regions"][name] = {"prediction": finish_metrics(predicted), "zero_reference": finish_metrics(reference)}
        return result
    target_power = response_view_to_range_power(
        arrays.response_view(view_index), pair_indices, device=device
    )
    target_intensity = normalize_power_db(
        target_power,
        float(stats["peak_power"]),
        float(stats["dynamic_range_db"]),
    )
    viewpoint = torch.as_tensor(arrays.viewpoint_positions[view_index], dtype=torch.float32, device=device)
    tx_pos = torch.as_tensor(arrays.tx_pos[view_index], dtype=torch.float32, device=device)
    rx_pos = torch.as_tensor(arrays.rx_pos[view_index], dtype=torch.float32, device=device)
    field = model.query_chunked(
        xyz,
        xyz - viewpoint[None, :],
        mask_progress=1.0,
        chunk_size=int(args.query_chunk),
    )
    pair_tensor = torch.as_tensor(pair_indices, dtype=torch.long, device=device)
    cell_values, cell_mass = bistatic_range_cells(
        torch.stack((field["alpha"], field["rcs"]), dim=-1),
        xyz,
        tx_pos,
        rx_pos,
        pair_tensor,
        bin_size=range_bin_size(arrays.metadata),
        num_bins=arrays.num_freq,
        pair_chunk=int(args.pair_chunk),
    )
    roi = scene_range_mask(ranges, viewpoint, float(args.extent), margin=float(args.range_margin))
    require(bool(roi.any().item()), f"view {view_index} has no scene-range ROI")
    valid_cells = cell_mass[:, roi] > 0
    pred_rcs = cell_values[:, roi, 1] * valid_cells.to(cell_values.dtype)
    target_roi = target_intensity[:, roi]
    roi_ranges = ranges[roi][None, :]
    prediction = radar_fields_intensity(
        pred_rcs,
        roi_ranges,
        offset=float(args.intensity_offset),
        scaler=float(args.intensity_scaler),
        range_law=str(args.range_law),
    )
    zero_prediction = radar_fields_intensity(
        torch.zeros_like(pred_rcs),
        roi_ranges,
        offset=float(args.intensity_offset),
        scaler=float(args.intensity_scaler),
        range_law=str(args.range_law),
    ).expand_as(prediction)
    whole = torch.ones_like(valid_cells, dtype=torch.bool)
    padded = ~valid_cells
    regions = {"whole_roi": whole, "valid_region": valid_cells, "padded_region": padded}
    result: Dict[str, Any] = {
        "view_id": int(view_index),
        "roi_range_bin_count": int(roi.sum().item()),
        "regions": {},
    }
    for name, mask in regions.items():
        predicted = metric_accumulator()
        zero = metric_accumulator()
        add_metrics(predicted, prediction, target_roi, mask)
        add_metrics(zero, zero_prediction, target_roi, mask)
        result["regions"][name] = {
            "prediction": finish_metrics(predicted),
            "zero_reference": finish_metrics(zero),
        }
    return result


def merge_role_results(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {"view_count": len(results), "regions": {}}
    for name in ("whole_roi", "valid_region", "padded_region"):
        merged_acc = {"prediction": metric_accumulator(), "zero_reference": metric_accumulator()}
        roi_bins = 0
        for result in results:
            roi_bins += int(result["roi_range_bin_count"])
            for source in ("prediction", "zero_reference"):
                metrics = result["regions"][name][source]
                if metrics is None:
                    continue
                count = float(metrics["measurement_count"])
                merged_acc[source]["count"] += count
                merged_acc[source]["abs_error"] += float(metrics["mean_abs_error"]) * count
                merged_acc[source]["sq_error"] += float(metrics["sq_error"])
                merged_acc[source]["target_power"] += float(metrics["target_power"])
        merged["regions"][name] = {
            "roi_range_bin_count_sum": roi_bins,
            "prediction": finish_metrics(merged_acc["prediction"]),
            "zero_reference": finish_metrics(merged_acc["zero_reference"]),
        }
    whole = merged["regions"]["whole_roi"]["prediction"]
    valid = merged["regions"]["valid_region"]["prediction"]
    padded = merged["regions"]["padded_region"]["prediction"]
    if whole and valid and padded:
        merged["padded_measurement_fraction"] = padded["measurement_count"] / whole["measurement_count"]
    else:
        merged["padded_measurement_fraction"] = None
    return merged


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--object")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/RIFT_dataset")
    parser.add_argument("--npz-path", type=Path)
    parser.add_argument("--sealed-split-manifest", "--role-manifest", type=Path)
    parser.add_argument("--allow-reserved-test", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--roles", nargs="+", choices=("train", "val", "validation", "test", "reserved_test"))
    parser.add_argument("--max-views", type=int, default=0, help="0 evaluates all selected role views")
    args = parser.parse_args(argv)
    from rift.rift_dataset import collection_manifest, resolve_object_inputs
    args.collection_mode = args.object is not None or (
        args.sealed_split_manifest is not None and collection_manifest(args.sealed_split_manifest))
    if args.collection_mode and args.config is None:
        parser.error("collection readout requires an explicit --config for its training recipe")
    args.config = args.config or DEFAULT_CONFIG
    args.npz_path, args.sealed_split_manifest = resolve_object_inputs(
        object_name=args.object, dataset_root=args.dataset_root,
        npz_path=args.npz_path, role_manifest_path=args.sealed_split_manifest)
    args.roles = [{"validation": "val", "reserved_test": "test"}.get(role, role)
                  for role in (args.roles or (["val"] if args.collection_mode else ["val", "test"]))]
    if len(set(args.roles)) != len(args.roles):
        parser.error("evaluation roles must be unique")
    if args.collection_mode and "test" in args.roles and not args.allow_reserved_test:
        parser.error("collection test evaluation requires --allow-reserved-test")
    return args


def collection_readout_inputs(args, checkpoint):
    from rift.rift_dataset import (load_object_contract, evaluation_role_indices,
                                   validate_checkpoint_object, object_identity)
    public, contract = load_object_contract(args.npz_path, args.sealed_split_manifest,
        response_roles=tuple(args.roles), allow_reserved_test=args.allow_reserved_test)
    if args.object is not None:
        validate_checkpoint_object(object_identity(args.object), contract)
    validate_checkpoint_object(checkpoint, contract)
    validate_checkpoint_object(checkpoint.get("power_stats", {}), contract)
    from rift.radar_fields_dataset import from_collection_arrays, validate_power_stats_acquisition
    arrays = from_collection_arrays(public, contract)
    validate_power_stats_acquisition(checkpoint.get('power_stats', {}), arrays.acquisition_identity)
    if checkpoint.get('sealed_protocol_contract', {}).get('acquisition_identity') != arrays.acquisition_identity:
        raise ValueError('Readout antenna acquisition differs from checkpoint')
    args.collection_arrays = arrays
    # Membership metadata is safe to retain for unrequested roles; the reader
    # capability above is restricted to the explicitly selected roles only.
    roles = {"train": contract["role_ids"]["train"], "val": contract["role_ids"]["validation"],
             "test": contract["role_ids"]["reserved_test"]}
    for role in args.roles:
        evaluation_role_indices(contract, role, allow_reserved_test=args.allow_reserved_test)
    return contract, roles


def main(argv=None) -> None:
    args = parse_args(argv)
    require(args.max_views >= 0, "--max-views must be nonnegative")
    config = load_json(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    require(isinstance(checkpoint, Mapping), "checkpoint must contain a mapping")
    collection_contract = None
    if args.collection_mode:
        collection_contract, collection_roles = collection_readout_inputs(args, checkpoint)
    elif "dataset_identity" in checkpoint.get("sealed_protocol_contract", {}):
        raise ValueError("A collection checkpoint requires its object-bound manifest")
    validate_checkpoint(
        checkpoint,
        config,
        npz_path=str(args.npz_path),
        manifest_path=str(args.sealed_split_manifest),
    )
    device = torch.device(args.device)
    require(device.type == "cuda", "native RF readout requires CUDA for the production package")
    require(torch.cuda.is_available(), "native RF readout requires the allocated CUDA device")
    require(not args.output.exists(), f"refusing to overwrite existing readout: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    saved_args = SimpleNamespace(**dict(checkpoint["args"]))
    from train_radar_fields import build_model
    model = build_model(saved_args, device)
    model.load_state_dict(checkpoint["radar_fields_state_dict"])
    model.eval()
    if recipe_name(saved_args) == "source-adapted-v3":
        torch.manual_seed(0)  # same fixed validation ray stream as the trainer

    arrays = (args.collection_arrays if collection_contract is not None
              else load_radar_fields_npz(str(args.npz_path), load_response=False))
    current_dataset = dataset_provenance(arrays)
    saved_dataset = checkpoint.get("dataset_provenance")
    require(isinstance(saved_dataset, Mapping), "checkpoint lacks dataset provenance")
    for key in ("response_shape", "response_dtype", "viewpoint_count", "tx_count", "rx_count", "frequency_count"):
        require(
            saved_dataset.get(key) == current_dataset.get(key),
            f"checkpoint dataset {key}={saved_dataset.get(key)!r} disagrees with readout dataset {current_dataset.get(key)!r}",
        )
    for key in (() if collection_contract else ("dataset_path", "dataset_file_size_bytes")):
        if saved_dataset.get(key) is not None and current_dataset.get(key) is not None:
            require(
                saved_dataset[key] == current_dataset[key],
                f"checkpoint dataset {key}={saved_dataset[key]!r} disagrees with readout dataset {current_dataset[key]!r}",
            )
    dataset_identity_verified = True
    split = load_radar_fields_sealed_split_manifest(
        str(args.sealed_split_manifest),
        arrays.num_views,
        response_shape=arrays._response_shape(),
        response_dtype=arrays.response_dtype,
        expected_num_train=len(collection_roles["train"]) if collection_contract else 3200,
        expected_num_val=1000,
        expected_num_test=1000,
    )
    role_ids = {"train": list(split.train_indices), "val": list(split.validation_indices), "test": list(split.test_indices)}
    if collection_contract is not None:
        require(role_ids == collection_roles, "split loader disagrees with registered object roles")
        role_ids = collection_roles
    checkpoint_roles = checkpoint["split_provenance"]["role_ids"]
    require(
        all(list(checkpoint_roles[name]) == role_ids[name] for name in ("train", "val", "test")),
        "readout manifest roles disagree with the training checkpoint",
    )
    selected_ids: list[int] = []
    for role in args.roles:
        selected_ids.extend(role_ids[role])
    if args.max_views:
        selected_ids = selected_ids[: args.max_views]
    require(selected_ids, "readout selected no views")
    arrays = restrict_radar_fields_response_views(arrays, selected_ids)

    xyz = generate_dynamic_grid(int(saved_args.granularity), float(saved_args.extent), device, jitter=False).reshape(-1, 3)
    ranges = range_bin_centers(arrays.metadata, device=device,
                              dtype=torch.float64 if native_recipe(saved_args) else torch.float32)
    stats = checkpoint["power_stats"]
    role_outputs: Dict[str, Any] = {}
    with torch.no_grad():
        for role in args.roles:
            if recipe_name(saved_args) == "source-adapted-v3":
                torch.manual_seed(0)
            role_selected = role_ids[role]
            if args.max_views:
                consumed = sum(min(len(role_ids[item]), args.max_views) for item in args.roles[: args.roles.index(role)])
                remaining = max(args.max_views - consumed, 0)
                role_selected = role_selected[:remaining]
            if not role_selected:
                continue
            results = [
                evaluate_view(
                    model,
                    arrays,
                    int(view_id),
                    xyz=xyz,
                    ranges=ranges,
                    stats=stats,
                    args=saved_args,
                    device=device,
                )
                for view_id in role_selected
            ]
            role_outputs[role] = merge_role_results(results)
    if device.type == "cuda":
        torch.cuda.synchronize()

    gpu_peak = int(torch.cuda.max_memory_allocated())
    gpu_total = int(torch.cuda.get_device_properties(0).total_memory)
    host_peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    host_budget = int(float(config["resources"]["memory_gb"]) * 1024**3)
    resource_report = {
        "scope": "native readout process only",
        "host_peak_rss_bytes": host_peak,
        "host_budget_bytes": host_budget,
        "gpu_peak_allocated_bytes": gpu_peak,
        "gpu_total_bytes": gpu_total,
        "headroom_fraction": 0.8,
        "host_headroom_pass": host_peak < 0.8 * host_budget,
        "gpu_headroom_pass": gpu_peak < 0.8 * gpu_total,
        "headroom_gate_pass": host_peak < 0.8 * host_budget and gpu_peak < 0.8 * gpu_total,
    }
    resource_path = args.output.parent / "readout_resource.json"
    require(not resource_path.exists(), f"refusing to overwrite existing resource report: {resource_path}")
    with resource_path.open("x", encoding="utf-8") as handle:
        json.dump(resource_report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    require(resource_report["headroom_gate_pass"], "RF native readout exceeded the 80 percent resource headroom gate")

    zero_value = float(
        radar_fields_intensity(
            torch.zeros(1, device=device),
            torch.ones(1, device=device),
            offset=float(saved_args.intensity_offset),
            scaler=float(saved_args.intensity_scaler),
            range_law=str(saved_args.range_law),
        ).item()
    )
    output = {
        "artifact_schema": "radar_fields_native_readout_v1",
        "radar_fields_recipe": recipe_contract(saved_args),
        "status": "measurement_only",
        "metric_domain": NORMALIZED_DB_INTENSITY_DOMAIN,
        "metric_label": NORMALIZED_DB_INTENSITY_LABEL,
        "reported_rel_mse_label": NORMALIZED_DB_RELMSE_LABEL,
        "checkpoint_step": int(checkpoint["step"]),
        "roles": list(role_outputs),
        "pair_count": int(arrays.num_tx * arrays.num_rx),
        "frequency_count": int(arrays.num_freq),
        "dataset_identity_verified": dataset_identity_verified,
        "dataset_identity": collection_contract["dataset_identity"] if collection_contract else None,
        "reserved_test_accessed": "test" in role_outputs,
        "zero_rcs_native_output": zero_value,
        "zero_reference_definition": "zero RCS passed through the same released native intensity transform",
        "resource_usage": resource_report,
        "regions": {
            "whole_roi": "all scene-range-mask entries, including unsupported/padded cells",
            "valid_region": ("entries with at least one ray sample inside registered support"
                             if native_recipe(saved_args) else
                             "entries where differentiable range-cell interpolation mass is positive"),
            "padded_region": "whole ROI minus valid region",
        },
        "objective_changed": False,
        "role_results": role_outputs,
    }
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"roles": list(role_outputs), "output": str(args.output)}, sort_keys=True))
    print("RF_B7873200_NATIVE_READOUT_PASS")


if __name__ == "__main__":
    main()
