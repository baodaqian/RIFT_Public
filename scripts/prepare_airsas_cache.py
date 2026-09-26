#!/usr/bin/env python3
"""Prepare a measured AirSAS cache for the shared RIFT/SH-SAS renderer.

This is intentionally a thin adapter around Reed's trusted deconvolution
outputs.  It does not reimplement the frontend or silently accept the
simulation dimensions.  Strict train-only caches are accepted only when the
frontend commandline metadata proves that Reed applied a positive supplied
scale to the raw traces and processed the first 43,200 transmissions.
Existing Reed globally normalized directories are rejected for this fixed comparison.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rift.airsas_contract import (  # noqa: E402
    EXPECTED_BANDWIDTH_KHZ,
    validate_system_data_5k,
)


WEIGHT_RE = re.compile(r"^weights_trans_(\d+)_(\d+)\.npy$")
RING_SIZE = 360
NUM_RINGS = 120


def _schema_dict(value: Any) -> Dict[str, Any]:
    raw = value._data if hasattr(value, "_data") else value
    if not isinstance(raw, dict):
        raise TypeError(f"expected schema dictionary, got {type(value)!r}")
    return raw


def _load_system_data(path: Path, reed_root: Path) -> Dict[str, Any]:
    sys.path.insert(0, str(reed_root.resolve()))
    from rift.sas_dataset import restricted_load_system_data

    return restricted_load_system_data(path)


def _weight_files(root: Path, stop: int) -> List[Tuple[int, int, Path]]:
    found = []
    for path in root.glob("weights_trans_*.npy"):
        match = WEIGHT_RE.match(path.name)
        if match is None:
            continue
        start, end = (int(match.group(1)), int(match.group(2)))
        if start <= stop - 1 and end >= start:
            found.append((start, end, path))
    found.sort(key=lambda item: (item[0], item[1]))
    expected = 0
    selected = []
    for start, end, path in found:
        if start != expected:
            if start > expected:
                break
            raise RuntimeError(f"overlapping or duplicate weight coverage at {path}")
        if end < start or end - start + 1 != RING_SIZE:
            raise RuntimeError(f"weight file {path} is not one complete 360-ping batch")
        selected.append((start, end, path))
        expected = end + 1
        if expected >= stop:
            break
    if expected != stop:
        raise RuntimeError(
            f"weight coverage is not exactly contiguous through row {stop - 1}; stopped at {expected - 1}"
        )
    return selected


def _analytic_signal(values: np.ndarray) -> np.ndarray:
    """Vectorized equivalent of Reed ``sas_utils.hilbert_torch``."""
    if values.ndim != 2:
        raise ValueError("deconvolved weights must have shape [ping, sample]")
    n = values.shape[-1]
    spectrum = np.fft.fft(values.astype(np.float64, copy=False), axis=-1)
    multiplier = np.zeros(n, dtype=np.float64)
    multiplier[0] = 1.0
    if n % 2 == 0:
        multiplier[n // 2] = 1.0
        multiplier[1 : n // 2] = 2.0
    else:
        multiplier[1 : (n + 1) // 2] = 2.0
    return np.fft.ifft(spectrum * multiplier[None, :], axis=-1).astype(np.complex64)


def _read_real_weights(files: Iterable[Tuple[int, int, Path]], num_bins: int) -> np.ndarray:
    blocks = []
    for start, end, path in files:
        values = np.load(path, allow_pickle=False)
        if values.shape != (end - start + 1, num_bins):
            raise RuntimeError(f"{path} has shape {values.shape}; expected {(end - start + 1, num_bins)}")
        if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
            raise RuntimeError(f"{path} contains non-finite or non-numeric deconvolved weights")
        blocks.append(np.asarray(values, dtype=np.float32))
    return np.concatenate(blocks, axis=0)


def _verify_rings(tx: np.ndarray, rx: np.ndarray, ring_size: int, num_rings: int) -> np.ndarray:
    expected = ring_size * num_rings
    if tx.shape != (expected, 3) or rx.shape != tx.shape:
        raise RuntimeError(f"expected {expected} complete rows for the first {num_rings} rings; got {tx.shape}")
    if not np.isfinite(tx).all() or not np.isfinite(rx).all():
        raise RuntimeError("measured Tx/Rx coordinates contain non-finite values")
    elevation = []
    for ring in range(num_rings):
        sl = slice(ring * ring_size, (ring + 1) * ring_size)
        # AirSAS's official acquisition is a turntable about its z axis.  We
        # verify rather than assume: every block must span a complete aperture
        # and its elevation must be internally stable.
        center = tx[sl].mean(axis=0)
        angles = np.unwrap(np.arctan2(tx[sl, 1] - center[1], tx[sl, 0] - center[0]))
        if abs(angles[-1] - angles[0]) < 5.0:
            raise RuntimeError(f"ring {ring} does not span a complete azimuth aperture")
        if float(np.ptp(tx[sl, 2])) > max(1.0e-5, 1.0e-3 * float(np.ptp(tx[:, 2]))):
            raise RuntimeError(f"ring {ring} is not a constant-elevation 360-ping block")
        elevation.append(float(tx[sl, 2].mean()))
    elevation = np.asarray(elevation)
    differences = np.diff(elevation)
    if not (np.all(differences > 0) or np.all(differences < 0)):
        raise RuntimeError("first 120 acquisition blocks are not ordered monotonically by elevation")
    return elevation


def _split_indices(num_pings: int, ring_size: int, num_rings: int):
    ring_ids = np.arange(num_rings, dtype=np.int64)
    test_rings = ring_ids[ring_ids % 10 == 2]
    validation_rings = ring_ids[ring_ids % 10 == 7]
    train_rings = np.setdiff1d(ring_ids, np.concatenate((test_rings, validation_rings)))
    def rows(rings):
        return np.concatenate([np.arange(r * ring_size, (r + 1) * ring_size) for r in rings]).astype(np.int64)
    return rows(train_rings), rows(validation_rings), rows(test_rings), train_rings, validation_rings, test_rings


def _normalization_contract(weights_dir: Path, requested: str) -> Tuple[str, Dict[str, Any]]:
    # Reed writes commandline_args.txt beside the numpy directory, while the
    # cache adapter receives that numpy directory as weights_dir.  Keep the
    # in-directory fallback for hand-produced smoke fixtures.
    candidates = (weights_dir.parent / "commandline_args.txt", weights_dir / "commandline_args.txt")
    args_path = next((path for path in candidates if path.exists()), candidates[0])
    args = None
    if args_path.exists():
        try:
            args = json.loads(args_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            raise RuntimeError(f"invalid frontend metadata JSON: {args_path}")

    recorded_mode = args.get("normalization_mode") if isinstance(args, dict) else None
    normalize_each = args.get("normalize_each") if isinstance(args, dict) else None
    if requested == "train_only" or (requested == "auto" and recorded_mode == "train_only"):
        if not isinstance(args, dict):
            raise RuntimeError("train_only requires commandline_args.txt from the corrected Reed frontend")
        if recorded_mode != "train_only":
            raise RuntimeError(
                "requested train_only but frontend metadata does not record normalization_mode=train_only"
            )
        if normalize_each is True:
            raise RuntimeError("per-waveform normalization is not allowed for train_only comparison")
        scale = args.get("normalization_scale")
        max_transmissions = args.get("max_transmissions")
        if isinstance(scale, bool) or scale is None:
            raise RuntimeError("train_only frontend metadata lacks normalization_scale")
        try:
            scale = float(scale)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("train_only normalization_scale is not numeric") from exc
        if not math.isfinite(scale) or scale <= 0.0:
            raise RuntimeError("train_only normalization_scale must be finite and positive")
        if isinstance(max_transmissions, bool) or not isinstance(max_transmissions, (int, float)):
            raise RuntimeError("train_only max_transmissions is not an integer")
        if not math.isfinite(float(max_transmissions)) or float(max_transmissions) != math.trunc(float(max_transmissions)):
            raise RuntimeError("train_only max_transmissions is not an integer")
        max_transmissions_int = int(max_transmissions)
        if max_transmissions_int != NUM_RINGS * RING_SIZE:
            raise RuntimeError("train_only frontend metadata must declare max_transmissions=43200")
        return "train_only", {
            "normalization_scale": scale,
            "max_transmissions": max_transmissions_int,
            "metadata_path": str(args_path),
        }

    if requested == "reed_global_max" and recorded_mode == "train_only":
        raise RuntimeError("frontend metadata proves train_only; do not relabel it as reed_global_max")
    if requested == "train_only":
        raise RuntimeError("requested train_only but frontend metadata does not prove the corrected Reed path")
    if isinstance(args, dict) and args.get("normalize_each") is True:
        raise RuntimeError("per-waveform normalization is not allowed for this comparison")
    if requested not in {"auto", "reed_global_max"}:
        raise ValueError(f"unsupported normalization mode {requested!r}")
    return "reed_global_max", {"metadata_path": str(args_path) if args_path.exists() else None}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("armadillo", "bunny"), required=True)
    parser.add_argument("--system-data", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reed-root", type=Path, default=Path("external/Reed_SAS_reference"))
    parser.add_argument("--bandwidth-khz", type=int, choices=(5,), default=5)
    parser.add_argument("--normalization-mode", choices=("train_only",), default="train_only")
    parser.add_argument("--max-rings", type=int, default=NUM_RINGS)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.max_rings != NUM_RINGS:
        raise ValueError("the first comparison is fixed to the predeclared 120-ring subset")
    if args.normalization_mode != "train_only":
        raise ValueError("the AirSAS5k comparison requires train_only normalization")
    output = args.output
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")).get("complete") is True:
        raise RuntimeError(f"complete cache already exists at {output}; use a new output identity")

    system = _load_system_data(args.system_data, args.reed_root)
    try:
        system_contract = validate_system_data_5k(args.system_data, system, scene=args.scene)
    except ValueError as exc:
        raise RuntimeError(f"not a supported {args.scene} AirSAS5k system-data package: {exc}") from exc
    if float(args.bandwidth_khz) != EXPECTED_BANDWIDTH_KHZ:
        raise RuntimeError(f"only {EXPECTED_BANDWIDTH_KHZ:g} kHz is supported for this comparison")
    weights_identity = args.weights_dir.as_posix().lower()
    if "20k" in weights_identity or "5k" not in weights_identity:
        raise RuntimeError("weights-dir must carry a 5k identity and cannot point at 20k weights")
    tx = np.asarray(system["tx_coords"], dtype=np.float32)
    rx = np.asarray(system["rx_coords"], dtype=np.float32)
    tx_vecs = None
    if "tx_vecs" in system and system["tx_vecs"] is not None:
        tx_vecs = np.asarray(system["tx_vecs"], dtype=np.float32)
        if tx_vecs.ndim != 2 or tx_vecs.shape[1:] != (3,) or tx_vecs.shape[0] < NUM_RINGS * RING_SIZE:
            raise RuntimeError("system-data tx_vecs must have shape [at least 43200,3]")
        if not np.isfinite(tx_vecs[: NUM_RINGS * RING_SIZE]).all():
            raise RuntimeError("system-data tx_vecs contains non-finite values in the first 43200 rows")
    crop = _schema_dict(system["crop_settings"])
    params = _schema_dict(system["sys_params"])
    geometry = _schema_dict(system["geometry"])
    num_pings = NUM_RINGS * RING_SIZE
    if tx.shape[0] < num_pings:
        raise RuntimeError(f"system data has only {tx.shape[0]} rows; 120 complete rings are required")
    elevation = _verify_rings(tx[:num_pings], rx[:num_pings], RING_SIZE, NUM_RINGS)

    files = _weight_files(args.weights_dir, num_pings)
    first = np.load(files[0][2], mmap_mode="r", allow_pickle=False)
    original_bins = int(first.shape[1])
    real_weights = _read_real_weights(files, original_bins)
    mode, frontend_contract = _normalization_contract(args.weights_dir, args.normalization_mode)
    if mode != "train_only":
        raise RuntimeError("the AirSAS5k comparison only accepts train_only normalization")

    train, validation, test, train_rings, validation_rings, test_rings = _split_indices(
        num_pings, RING_SIZE, NUM_RINGS
    )
    weights = _analytic_signal(real_weights)

    # Import Reed's crop helper rather than reproducing its padding/rounding.
    sys.path.insert(0, str(args.reed_root.resolve()))
    from sas_utils import crop_wfm
    from inr_reconstruction.utils import aggressive_crop_weights
    import torch

    waveform = np.asarray(system["wfm"])
    fs = float(params["sampling_frequency"])
    speed = float(system["speed_of_sound"])
    computed_crop = _schema_dict(
        crop_wfm(tx, rx, np.asarray(geometry["corners"]), waveform.shape[-1], fs, speed)
    )
    for key in ("min_dist", "max_dist", "num_samples"):
        if key not in crop or key not in computed_crop:
            raise RuntimeError(f"crop metadata is missing {key}")
        if key == "num_samples":
            if int(crop[key]) != int(computed_crop[key]):
                raise RuntimeError(f"stored and Reed-computed crop disagree for {key}")
        elif not np.isclose(float(crop[key]), float(computed_crop[key]), rtol=0.0, atol=1.0e-6):
            raise RuntimeError(f"stored and Reed-computed crop disagree for {key}")
    if int(crop["num_samples"]) != original_bins or original_bins != system_contract["crop_bins"]:
        raise RuntimeError(
            f"{args.scene} AirSAS5k weight bins {original_bins} do not match official crop bins {crop['num_samples']}"
        )

    old_min_dist = float(crop["min_dist"])
    old_max_dist = float(crop["max_dist"])
    new_min_sample, new_max_sample = aggressive_crop_weights(
        tx_coords=torch.from_numpy(tx[:num_pings]),
        rx_coords=torch.from_numpy(rx[:num_pings]),
        corners=torch.from_numpy(np.asarray(geometry["corners"], dtype=np.float32)),
        old_min_dist=old_min_dist,
        old_max_dist=old_max_dist,
        num_radial=original_bins,
    )
    new_min_sample = int(new_min_sample)
    new_max_sample = int(new_max_sample)
    if not (0 <= new_min_sample < new_max_sample <= original_bins):
        raise RuntimeError(
            f"Reed aggressive crop returned invalid [{new_min_sample}:{new_max_sample}] "
            f"for {original_bins} original bins"
        )
    full_radii = np.linspace(old_min_dist, old_max_dist, original_bins, dtype=np.float32)
    weights = weights[:, new_min_sample:new_max_sample]
    radii = full_radii[new_min_sample:new_max_sample]
    num_bins = int(weights.shape[1])

    output.mkdir(parents=True, exist_ok=True)
    geometry_payload = dict(
        tx_coords=tx[:num_pings],
        rx_coords=rx[:num_pings],
        radii=radii,
        corners=np.asarray(geometry["corners"], dtype=np.float32),
        voxels=np.asarray(geometry["voxels"], dtype=np.float32),
        grid_shape=np.asarray([geometry["num_x"], geometry["num_y"], geometry["num_z"]], dtype=np.int64),
        source_ids=np.arange(num_pings, dtype=np.int64),
        train_indices=train,
        validation_indices=validation,
        test_indices=test,
        elevation_ring=np.repeat(np.arange(NUM_RINGS, dtype=np.int64), RING_SIZE),
    )
    if tx_vecs is not None:
        geometry_payload["tx_vecs"] = tx_vecs[:num_pings]
    np.savez(output / "geometry.npz", **geometry_payload)
    np.save(output / "weights.npy", weights)
    manifest = {
        "complete": True,
        "cache_contract_version": 2,
        "dataset_identity": system_contract["identity"],
        "frontend_asset_identity": system_contract["frontend_asset_identity"],
        "scene": args.scene,
        "bandwidth_khz": float(args.bandwidth_khz),
        "frontend": "reed_neural_pulse_deconvolution_outputs",
        "weights_source": str(args.weights_dir.resolve()),
        "system_data_source": str(args.system_data.resolve()),
        "num_pings": num_pings,
        "num_bins": num_bins,
        "original_num_bins": original_bins,
        "ring_size": RING_SIZE,
        "num_rings": NUM_RINGS,
        "ring_protocol": "first_120_complete_elevation_rings",
        "held_out_protocol": "elevation_interpolation_within_measured_aperture",
        "split_contract": {
            "explicit": True,
            "train_indices": train.tolist(),
            "validation_indices": validation.tolist(),
            "test_indices": test.tolist(),
            "train_ring_residue":  "all except 2 and 7 mod 10",
            "validation_ring_residue": 7,
            "test_ring_residue": 2,
            "counts": {"train_rings": int(train_rings.size), "validation_rings": int(validation_rings.size), "test_rings": int(test_rings.size)},
        },
        "frontend_normalization": {
            "mode": mode,
            "strict_train_only": mode == "train_only",
            "applied_by": "reed_deconvolver_raw_trace_stage" if mode == "train_only" else None,
            "supplied_scale": frontend_contract.get("normalization_scale"),
            "max_transmissions": frontend_contract.get("max_transmissions"),
            "metadata_path": frontend_contract.get("metadata_path"),
            "fit_definition": "scale supplied from cropped raw TRAIN rows within the first 43200 pings",
            "global_normalization_allowed_only_for_smoke": mode == "reed_global_max",
        },
        "aggressive_crop": {
            "helper": "Reed_SAS_reference/inr_reconstruction/utils.py::aggressive_crop_weights",
            "original_min_dist": old_min_dist,
            "original_max_dist": old_max_dist,
            "new_min_sample": new_min_sample,
            "new_max_sample_exclusive": new_max_sample,
            "cropped_min_dist": float(radii[0]),
            "cropped_max_dist": float(radii[-1]),
        },
        "sound_speed_mps": speed,
        "sample_rate_hz": fs,
        "crop": {key: float(computed_crop[key]) if key != "num_samples" else int(computed_crop[key]) for key in ("min_sample", "min_dist", "max_dist", "num_samples") if key in computed_crop},
        "geometry_grid_shape": [int(geometry["num_x"]), int(geometry["num_y"]), int(geometry["num_z"])],
        "system_data_contract": system_contract,
        "elevation_values": elevation.tolist(),
        "source_ids_are_row_ids": True,
        "tx_direction": {
            "source": "system_data.tx_vecs" if tx_vecs is not None else "point_at_center_fallback",
            "present": tx_vecs is not None,
            "shape": [int(tx_vecs[:num_pings].shape[0]), 3] if tx_vecs is not None else None,
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote measured AirSAS cache: {output}")
    print(f"split train/validation/test={train.size}/{validation.size}/{test.size}; bins={num_bins}; normalization={mode}")


if __name__ == "__main__":
    main()
