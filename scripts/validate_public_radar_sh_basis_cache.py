#!/usr/bin/env python
"""CPU contract gates for the shared PublicRadar directional-SH cache."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_cache_module():
    path = PROJECT_ROOT / "rift" / "public_radar_sh_cache.py"
    spec = importlib.util.spec_from_file_location(
        "rift_public_radar_sh_cache_validation", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def gate(name, passed):
    if not bool(passed):
        raise AssertionError(name)
    print(f"PASS: {name}", flush=True)


class FakeDataset:
    def __init__(self, theta, phi, canonical_indices):
        self.items = [
            (
                torch.arange(3),
                torch.tensor([phi[view_id]], dtype=torch.float32),
                torch.tensor([theta[view_id]], dtype=torch.float32),
                torch.ones(3, 1),
                torch.zeros(3, 1),
                torch.zeros(1, 3),
                torch.zeros(1, 3),
            )
            for view_id in canonical_indices
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class AuthoritativeDirectionDataset:
    """Direction-only surrogate for PecSphereNPZDataset's scalar item path."""

    def __init__(self, viewpoint_positions, canonical_indices):
        self.positions = np.asarray(viewpoint_positions, dtype=np.float64)
        self.canonical_indices = np.asarray(canonical_indices, dtype=np.int64)
        self._dummy = torch.zeros(1)

    def __len__(self):
        return self.canonical_indices.size

    def __getitem__(self, index):
        view_id = int(self.canonical_indices[index])
        position = self.positions[view_id]
        radius = float(np.linalg.norm(position))
        theta = float(
            np.arccos(np.clip(position[2] / radius, -1.0, 1.0))
        )
        phi = float(np.arctan2(position[1], position[0]))
        return (
            self._dummy,
            torch.tensor([phi], dtype=torch.float32),
            torch.tensor([theta], dtype=torch.float32),
            self._dummy,
            self._dummy,
            self._dummy,
            self._dummy,
        )


def expect_failure(exception, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except exception:
        return True
    return False


def synthetic_gate(cache_module):
    rng = np.random.default_rng(17)
    positions = rng.normal(size=(23, 3)).astype(np.float64)
    positions[:, 2] += 2.0
    with tempfile.TemporaryDirectory(prefix="rift_sh_cache_gate_") as directory:
        cache_dir = Path(directory) / "cache"
        manifest = cache_module.materialize_cache(
            cache_dir, positions, "synthetic_scene"
        )
        gate("cache materializes degree 6", manifest["max_degree"] == 6)
        cache = cache_module.PublicRadarSHBasisCache(
            cache_dir, positions, "synthetic_scene"
        )
        theta, phi = cache_module.canonical_theta_phi(positions)
        order = np.asarray([11, 2, 11, 22, 0], dtype=np.int64)
        for degree in (0, 3, 6):
            actual = cache.basis_rows(
                order, degree, torch.device("cpu"), dtype=torch.float32
            )
            expected = torch.from_numpy(
                cache_module.basis_from_angles(theta[order], phi[order], degree)
            ).transpose(0, 1).contiguous()
            relative_error = float(
                torch.linalg.vector_norm(actual - expected)
                / torch.linalg.vector_norm(expected).clamp_min(1.0e-30)
            )
            print(
                f"INFO: cached degree-{degree} relative_error={relative_error:.9g}",
                flush=True,
            )
            gate(
                f"degree-{degree} cache preserves order/repeats/prefix",
                actual.shape == ((degree + 1) ** 2, order.size)
                and relative_error <= 2.0e-6,
            )
        second = cache_module.materialize_cache(
            cache_dir, positions, "synthetic_scene"
        )
        gate("existing compatible cache is validated, not overwritten", second == manifest)
        changed = positions.copy()
        changed[3, 0] += 0.1
        gate(
            "changed viewpoint directions invalidate the cache",
            expect_failure(
                ValueError,
                cache_module.PublicRadarSHBasisCache,
                cache_dir,
                changed,
                "synthetic_scene",
            ),
        )
        gate(
            "changed dataset identity invalidates the cache",
            expect_failure(
                ValueError,
                cache_module.PublicRadarSHBasisCache,
                cache_dir,
                positions,
                "other_scene",
            ),
        )
        gate(
            "degree above six is rejected",
            expect_failure(
                ValueError,
                cache.basis_rows,
                order,
                7,
                torch.device("cpu"),
            ),
        )
        gate(
            "out-of-range view id is rejected",
            expect_failure(
                IndexError,
                cache.basis_rows,
                np.asarray([23]),
                3,
                torch.device("cpu"),
            ),
        )

        manifest_path = cache_dir / cache_module.MANIFEST_FILENAME
        semantic_fields = (
            "basis_order",
            "angle_source",
            "theta_convention",
            "phi_convention",
            "viewpoint_positions_source_dtype",
        )
        for field_name in semantic_fields:
            changed_manifest = dict(manifest)
            changed_manifest[field_name] = f"invalid_{field_name}"
            manifest_path.write_text(
                json.dumps(changed_manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            gate(
                f"changed {field_name} invalidates the cache",
                expect_failure(
                    ValueError,
                    cache_module.PublicRadarSHBasisCache,
                    cache_dir,
                    positions,
                    "synthetic_scene",
                ),
            )
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        del cache
        gc.collect()
        basis_path = cache_dir / cache_module.BASIS_FILENAME
        changed_basis = np.load(basis_path, mmap_mode="r+", allow_pickle=False)
        changed_basis[1, 5] += np.float32(1.0e-4)
        changed_basis.flush()
        del changed_basis
        gc.collect()
        gate(
            "finite corruption outside the former sparse canary is rejected",
            expect_failure(
                ValueError,
                cache_module.PublicRadarSHBasisCache,
                cache_dir,
                positions,
                "synthetic_scene",
            ),
        )

    synthetic_indices = np.asarray([19, 4, 8], dtype=np.int64)
    base = FakeDataset(theta, phi, synthetic_indices)
    indexed = cache_module.IndexedPublicRadarDataset(
        base, synthetic_indices, positions
    )
    gate("indexed dataset preserves length", len(indexed) == len(base))
    gate(
        "indexed dataset replaces only the unused frequency field",
        indexed[1][0] == 4
        and all(
            torch.equal(actual, expected)
            for actual, expected in zip(indexed[1][1:], base[1][1:])
        ),
    )
    gate(
        "item view ids preserve requested batch order",
        np.array_equal(
            cache_module.item_view_indices([indexed[2], indexed[0], indexed[2]]),
            np.asarray([8, 19, 8]),
        ),
    )
    gate(
        "wrong same-length canonical ordering is rejected",
        expect_failure(
            ValueError,
            cache_module.IndexedPublicRadarDataset,
            base,
            np.asarray([4, 19, 8], dtype=np.int64),
            positions,
        ),
    )


def real_cache_gate(cache_module, scene, npz_path, cache_dir):
    with np.load(npz_path, allow_pickle=False) as archive:
        positions = np.array(archive["viewpoint_positions"], copy=True)
        role_order = np.concatenate(
            [
                np.asarray(archive["train_indices"], dtype=np.int64),
                np.asarray(archive["validation_indices"], dtype=np.int64),
                np.asarray(archive["test_indices"], dtype=np.int64),
            ]
        )
    gate(
        f"{scene} role arrays exactly partition canonical view ids",
        role_order.size == positions.shape[0]
        and np.array_equal(
            np.sort(role_order), np.arange(positions.shape[0], dtype=np.int64)
        ),
    )
    cache = cache_module.PublicRadarSHBasisCache(cache_dir, positions, scene)
    sample = np.unique(
        np.linspace(0, positions.shape[0] - 1, 17, dtype=np.int64)
    )
    theta, phi = cache_module.canonical_theta_phi(positions)
    for degree in (0, 3, 6):
        actual = cache.basis_rows(sample, degree, torch.device("cpu"))
        expected = torch.from_numpy(
            cache_module.basis_from_angles(theta[sample], phi[sample], degree)
        ).transpose(0, 1).contiguous()
        relative_error = float(
            torch.linalg.vector_norm(actual - expected)
            / torch.linalg.vector_norm(expected).clamp_min(1.0e-30)
        )
        gate(
            f"{scene} real cache degree-{degree} canary",
            relative_error <= 2.0e-6,
        )

    base = AuthoritativeDirectionDataset(positions, role_order)
    indexed = cache_module.IndexedPublicRadarDataset(
        base, role_order, positions
    )
    probes = np.unique(
        np.linspace(0, role_order.size - 1, min(17, role_order.size), dtype=np.int64)
    )
    gate(
        f"{scene} role/permutation item ids preserve canonical order",
        np.array_equal(
            cache_module.item_view_indices([indexed[int(i)] for i in probes]),
            role_order[probes],
        ),
    )

    theta, phi = cache_module.canonical_theta_phi(positions)
    first_id = int(role_order[0])
    different = np.flatnonzero(
        (theta[role_order] != theta[first_id]) | (phi[role_order] != phi[first_id])
    )
    gate(f"{scene} contains independently distinguishable view directions", different.size > 0)
    wrong_order = role_order.copy()
    swap_at = int(different[0])
    wrong_order[0], wrong_order[swap_at] = wrong_order[swap_at], wrong_order[0]
    gate(
        f"{scene} wrong same-length global ordering is rejected",
        expect_failure(
            ValueError,
            cache_module.IndexedPublicRadarDataset,
            base,
            wrong_order,
            positions,
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--real-cache",
        action="append",
        nargs=3,
        metavar=("SCENE", "NPZ", "CACHE_DIR"),
        default=[],
    )
    args = parser.parse_args()
    cache_module = load_cache_module()
    synthetic_gate(cache_module)
    for scene, npz_path, cache_dir in args.real_cache:
        real_cache_gate(cache_module, scene, npz_path, cache_dir)
    print("PUBLIC_RADAR_SH_CACHE_VALIDATION_OK", flush=True)


if __name__ == "__main__":
    main()
