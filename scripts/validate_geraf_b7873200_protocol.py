#!/usr/bin/env python
"""Data-free regression checks for the corrected GeRaF B7873200 split gate.

This deliberately does not open the 12 GB B787 response archive.  The
allocated-node target-preparation gate is responsible for header ingress and
native matched-filter checks.  Here we prove that the semantic contract which
will be recorded in a GeRaF cache accepts only the fixed interpolation split
and never authorizes its test or unused roles.
"""

from __future__ import annotations

import importlib.util
import sys
import copy
from pathlib import Path
from typing import Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_PROTOCOL_SPEC = importlib.util.spec_from_file_location(
    "geraf_b7873200_protocol_under_test", REPO_ROOT / "rift" / "geraf_b7873200_protocol.py"
)
if _PROTOCOL_SPEC is None or _PROTOCOL_SPEC.loader is None:
    raise RuntimeError("cannot load the isolated GeRaF B7873200 protocol module")
protocol = importlib.util.module_from_spec(_PROTOCOL_SPEC)
sys.modules[_PROTOCOL_SPEC.name] = protocol
_PROTOCOL_SPEC.loader.exec_module(protocol)

B787_3200_CANONICAL_NPZ_PATH = protocol.B787_3200_CANONICAL_NPZ_PATH
B787_3200_ACQUISITION_SCHEMA = protocol.B787_3200_ACQUISITION_SCHEMA
B787_3200_CACHE_ACQUISITION_FILENAME = protocol.B787_3200_CACHE_ACQUISITION_FILENAME
B787_3200_CACHE_SCHEMA = protocol.B787_3200_CACHE_SCHEMA
B787_3200_MANIFEST_NAME = protocol.B787_3200_MANIFEST_NAME
B787_3200_NUM_TEST = protocol.B787_3200_NUM_TEST
B787_3200_NUM_TRAIN = protocol.B787_3200_NUM_TRAIN
B787_3200_NUM_UNUSED = protocol.B787_3200_NUM_UNUSED
B787_3200_NUM_VALIDATION = protocol.B787_3200_NUM_VALIDATION
B787_3200_NUM_VIEWS = protocol.B787_3200_NUM_VIEWS
B787_3200_SEED = protocol.B787_3200_SEED
b7873200_sealed_protocol_identity = protocol.b7873200_sealed_protocol_identity


def _valid_recipe(identity: dict[str, object]) -> dict[str, object]:
    return {
        "schema": B787_3200_CACHE_SCHEMA,
        "version": 1,
        "sealed_protocol_identity": copy.deepcopy(identity),
        "response_roles_materialized": ["train", "validation"],
        "acquisition_record": {
            "schema": B787_3200_ACQUISITION_SCHEMA,
            "filename": B787_3200_CACHE_ACQUISITION_FILENAME,
        },
        "target_spec": {
            "native_readout": "complex magnitude |MF|",
            "phase_sign": -1.0,
            "backend": "range",
            "compute_dtype": "float64",
            "grid": {
                "scene_extent_m": 0.15,
                "n_azimuth": 32,
                "n_elevation": 32,
                "n_depth": 32,
                "aperture_scale": 1.0,
            },
            "operator": {
                "implementation": "range_nufft",
                "range_model": "none",
                "include_four_pi": False,
                "freq_chunk": None,
                "kernel_width": 20,
                "oversample": 2,
                "point_chunk": 4096,
                "pair_chunk": 32,
            },
        },
    }


class Gates:
    def __init__(self) -> None:
        self.count = 0

    def check(self, condition: bool, detail: str) -> None:
        if not condition:
            raise AssertionError(detail)
        self.count += 1
        print(f"PASS {self.count:02d}: {detail}", flush=True)


def _canonical_contract() -> dict[str, object]:
    permutation = np.random.Generator(np.random.PCG64(B787_3200_SEED)).permutation(
        B787_3200_NUM_VIEWS
    )
    validation_start = B787_3200_NUM_VIEWS - B787_3200_NUM_VALIDATION
    test_start = validation_start - B787_3200_NUM_TEST
    train = permutation[:B787_3200_NUM_TRAIN]
    validation = permutation[validation_start:]
    test = permutation[test_start:validation_start]
    return {
        "schema": "rift_npz_sealed_protocol_v1",
        "version": 1,
        "data_format": "npz",
        # These two paths intentionally model the historical manifest hint;
        # the returned identity must remain path-free and permit the valid
        # /storage/home archive selected by the current recipe.
        "source_path": "/storage/home/example/b787.npz",
        "role_manifest_path": "/storage/project/retired/split.json",
        "response_shape": [B787_3200_NUM_VIEWS, 16, 16, 1, 600],
        "response_dtype": "complex64",
        "role_manifest_name": B787_3200_MANIFEST_NAME,
        "split_strategy": "fixed_tail_subsampled",
        "role_ids": {
            "train": train.tolist(),
            "validation": validation.tolist(),
            "reserved_test": test.tolist(),
            "unused": permutation[
                B787_3200_NUM_TRAIN : B787_3200_NUM_VIEWS
                - B787_3200_NUM_VALIDATION
                - B787_3200_NUM_TEST
            ].tolist(),
        },
        "response_access": {
            "train_materialized": True,
            "validation_materialized": True,
            "reserved_test_materialized": False,
            "unused_materialized": False,
        },
    }


def _expect_rejection(gates: Gates, detail: str, mutate: Callable[[dict[str, object]], None]) -> None:
    contract = _canonical_contract()
    mutate(contract)
    try:
        b7873200_sealed_protocol_identity(contract)
    except ValueError:
        gates.check(True, detail)
    else:
        raise AssertionError(detail)


def main() -> None:
    gates = Gates()
    contract = _canonical_contract()
    identity = b7873200_sealed_protocol_identity(contract)
    roles = identity["role_ids"]
    assert isinstance(roles, dict)
    gates.check(
        B787_3200_CANONICAL_NPZ_PATH
        == "/storage/home/hcoda1/1/dbao31/r-jromberg3-0/RIFT/data/"
        "b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz",
        "the corrected recipe names the canonical /storage/home B787 archive",
    )
    gates.check(
        [len(roles[name]) for name in ("train", "validation", "reserved_test", "unused")]
        == [B787_3200_NUM_TRAIN, B787_3200_NUM_VALIDATION, B787_3200_NUM_TEST, B787_3200_NUM_UNUSED],
        "the semantic identity preserves the 3200/1000/1000/4800 role sizes",
    )
    gates.check(
        len(set().union(*(set(roles[name]) for name in roles))) == B787_3200_NUM_VIEWS,
        "the semantic identity is a complete disjoint partition",
    )
    gates.check(
        identity["response_access"]
        == {
            "train_materialized": True,
            "validation_materialized": True,
            "reserved_test_materialized": False,
            "unused_materialized": False,
        },
        "only train and validation are authorized for response-derived targets",
    )
    gates.check(
        "source_path" not in identity and "role_manifest_path" not in identity,
        "archive and manifest locations remain provenance rather than cache identity",
    )
    _expect_rejection(
        gates,
        "a noncanonical manifest name is rejected",
        lambda item: item.__setitem__("role_manifest_name", "other_manifest"),
    )
    _expect_rejection(
        gates,
        "a reordered training role is rejected",
        lambda item: item["role_ids"]["train"].__setitem__(0, 0),  # type: ignore[index]
    )
    _expect_rejection(
        gates,
        "a test/train overlap is rejected",
        lambda item: item["role_ids"]["reserved_test"].__setitem__(0, item["role_ids"]["train"][0]),  # type: ignore[index]
    )
    _expect_rejection(
        gates,
        "a changed unused-role order is rejected",
        lambda item: item["role_ids"]["unused"].reverse(),  # type: ignore[index]
    )
    _expect_rejection(
        gates,
        "an attempt to mark reserved test materialized is rejected",
        lambda item: item["response_access"].__setitem__("reserved_test_materialized", True),  # type: ignore[index]
    )
    _expect_rejection(
        gates,
        "a response layout other than 16x16x1x600 is rejected",
        lambda item: item.__setitem__("response_shape", [B787_3200_NUM_VIEWS, 16, 16, 600]),
    )
    recipe = _valid_recipe(identity)
    protocol._validated_b7873200_recipe(recipe, identity)
    gates.check(
        True,
        "the corrected recipe binds the direct acquisition record and phase-only range operator",
    )
    for detail, mutate in (
        (
            "a cache recipe with a different range model is rejected",
            lambda item: item["target_spec"]["operator"].__setitem__("range_model", "physical"),
        ),
        (
            "a cache recipe with four-pi scaling is rejected",
            lambda item: item["target_spec"]["operator"].__setitem__("include_four_pi", True),
        ),
        (
            "a cache recipe with a direct frequency chunk is rejected",
            lambda item: item["target_spec"]["operator"].__setitem__("freq_chunk", 32),
        ),
        (
            "a cache recipe without the direct acquisition record is rejected",
            lambda item: item.__setitem__("acquisition_record", {}),
        ),
    ):
        altered = copy.deepcopy(recipe)
        mutate(altered)
        try:
            protocol._validated_b7873200_recipe(altered, identity)
        except ValueError:
            gates.check(True, detail)
        else:
            raise AssertionError(detail)
    print(f"GeRaF B7873200 sealed protocol passed: {gates.count} checks.", flush=True)


if __name__ == "__main__":
    main()
