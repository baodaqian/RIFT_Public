#!/usr/bin/env python3
"""Sealed finite-SH signal interpolation or train-only matched-filter geometry.

These are model-free baselines, not neural trainers. The historical angular
oracle implementation stays unchanged; only its fitting/basis functions are
reused with an explicitly restricted RIFT dataset reader.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from rift.npz_dataset import (build_freqs, get_npz_response_view,
                             iter_npz_response_views, restrict_npz_response_views)
from rift.rift_dataset import (DEFAULT_ROOT, collection_contract, load_object_contract, resolve_object_inputs,
                               validate_checkpoint_object, object_identity)
from rift.radarsplat_b7873200_protocol import atomic_write_json, atomic_save_npz
from scripts.eval_bandlimit_oracle import real_sh_basis, fit_and_score


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", help="RIFT dataset object or registered alias")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--npz-path")
    parser.add_argument("--role-manifest")
    parser.add_argument("--method", choices=("fsh", "mfbp"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    args.npz_path, args.role_manifest = resolve_object_inputs(
        object_name=args.object, dataset_root=args.dataset_root,
        npz_path=args.npz_path, role_manifest_path=args.role_manifest)
    return args


def finite_sh(arrays, contract, output):
    roles = contract["role_ids"]
    selected = roles["train"] + roles["validation"]
    arrays = restrict_npz_response_views(arrays, selected)
    row_for_id = {view: row for row, view in enumerate(selected)}
    # Match the historical oracle's published subsampling, explicitly labeled.
    block = np.empty((len(selected), 8 * 8 * 60), dtype=np.complex64)
    for view, response in iter_npz_response_views(arrays, selected):
        block[row_for_id[view]] = response.mean(axis=2)[::2, ::2, ::10].reshape(-1)
    positions = arrays["viewpoint_positions"][selected]
    basis = real_sh_basis(positions / np.linalg.norm(positions, axis=1, keepdims=True), 32)
    n_train = len(roles["train"])
    train_y, val_y = block[:n_train], block[n_train:]
    rows = []
    for degree in (0, 4, 8, 12, 16, 20, 24, 28, 32):
        columns = (degree + 1) ** 2
        train_error, val_error = fit_and_score(
            basis[:n_train, :columns], basis[n_train:, :columns], train_y, val_y, 0.0)
        rows.append({"degree": degree, "train_rel_mse": train_error, "val_rel_mse": val_error})
    atomic_write_json(output / "finite_sh_interpolator.json", {
        "sealed_protocol_identity": contract, "freq_stride": 10, "pair_stride": 2,
        "degree_sweep": rows, "best": min(rows, key=lambda row: row["val_rel_mse"]),
        "selection": "validation-selected oracle diagnostic; not a reserved-test score",
        "metric": "pooled complex relative MSE on subsampled channels/frequencies",
        "geometry_supported": False})


def matched_filter(arrays, contract, output, device, resume):
    from rift.matched_filter_power import matched_filter_complex
    roles = contract["role_ids"]
    arrays = restrict_npz_response_views(arrays, roles["train"])
    edges = np.linspace(-0.15, 0.15, 49)
    centers = (edges[:-1] + edges[1:]) * 0.5
    points = np.stack(np.meshgrid(centers, centers, centers, indexing="ij"), axis=-1).reshape(-1, 3)
    points = torch.as_tensor(points, dtype=torch.float64, device=device)
    freqs = torch.as_tensor(build_freqs(arrays["meta"]), dtype=torch.float64, device=device)
    accumulator = torch.zeros(len(points), dtype=torch.complex128, device=device)
    progress = output / "accumulator_latest.npz"
    identity_text = json.dumps(contract, sort_keys=True)
    offset = 0
    if progress.exists():
        if not resume:
            raise ValueError("Existing matched-filter accumulator requires --resume")
        with np.load(progress, allow_pickle=False) as saved:
            if str(saved["identity"].item()) != identity_text:
                raise ValueError("Matched-filter continuation would change its object/split")
            offset = int(saved["next_train_offset"])
            if not 0 <= offset <= len(roles["train"]) or saved["complex_adjoint"].shape != (48 ** 3,):
                raise ValueError("Invalid matched-filter accumulator")
            accumulator.copy_(torch.as_tensor(saved["complex_adjoint"], device=device))
    for index in range(offset, len(roles["train"])):
        view = roles["train"][index]
        response = get_npz_response_view(arrays, view).mean(axis=2)
        contribution = matched_filter_complex(
            torch.as_tensor(response, device=device),
            torch.as_tensor(arrays["tx_pos"][view], dtype=torch.float64, device=device),
            torch.as_tensor(arrays["rx_pos"][view], dtype=torch.float64, device=device),
            freqs, points, phase_sign=-1.0, response_layout="tx_rx_freq",
            range_model="none", include_four_pi=False, backend="range_nufft",
            point_chunk=16384, pair_chunk=32, nufft_kernel_width=20,
            nufft_oversample=2, compute_dtype=torch.float64)
        if not torch.isfinite(contribution).all():
            raise FloatingPointError(f"Non-finite matched filter at view {view}")
        accumulator += contribution
        if (index + 1) % 5 == 0 or index == 3199:
            atomic_save_npz(progress, complex_adjoint=accumulator.cpu().numpy(),
                            next_train_offset=np.asarray(index + 1), identity=np.asarray(identity_text))
    grid = accumulator.reshape(48, 48, 48).cpu().numpy()
    magnitude = np.abs(grid)
    span = np.ptp(magnitude)
    if not np.isfinite(span) or span <= 0:
        raise FloatingPointError("Matched-filter magnitude has no finite dynamic range")
    atomic_save_npz(output / "matched_filter.npz", complex_adjoint=grid,
                    magnitude=magnitude, normalized_magnitude=(magnitude - magnitude.min()) / span,
                    grid_centers=centers, extent=np.asarray(0.15), phase_sign=np.asarray(-1.0),
                    train_indices=np.asarray(roles["train"]), identity=np.asarray(identity_text))


def main(argv=None):
    args = parse_args(argv)
    arrays, identity = load_object_contract(args.npz_path, args.role_manifest,
        response_roles=("train", "validation") if args.method == "fsh" else ("train",))
    if args.object is not None:
        validate_checkpoint_object(object_identity(args.object), identity)
    identity = collection_contract(identity)  # Paths remain provenance, not resume identity.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "model_free_identity.json"
    expected = {"method": args.method, "sealed_protocol_identity": identity}
    if config_path.exists() and json.loads(config_path.read_text()) != expected:
        raise ValueError("Output directory already belongs to another object/method")
    atomic_write_json(config_path, expected)
    if args.method == "fsh":
        finite_sh(arrays, identity, args.output_dir)
    else:
        matched_filter(arrays, identity, args.output_dir, args.device, args.resume)


if __name__ == "__main__":
    main()
