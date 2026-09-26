#!/usr/bin/env python
"""Data-free contract checks for the sealed GeRaF B7873200 response adapter.

This intentionally imports neither torch nor the historical GeRaF/RIFT
trainer.  It creates only tiny temporary stored/compressed NPZ fixtures and
patches the adapter's two delayed live-runtime seams: the canonical protocol
preflight and stored-member mapper.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import zipfile
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_ADAPTER_SPEC = importlib.util.spec_from_file_location(
    "geraf_b7873200_adapter_under_test", ROOT / "rift" / "geraf_b7873200_adapter.py"
)
if _ADAPTER_SPEC is None or _ADAPTER_SPEC.loader is None:
    raise RuntimeError("cannot load the isolated GeRaF B7873200 adapter module")
adapter = importlib.util.module_from_spec(_ADAPTER_SPEC)
sys.modules[_ADAPTER_SPEC.name] = adapter
_ADAPTER_SPEC.loader.exec_module(adapter)


class Gates:
    def __init__(self) -> None:
        self.count = 0

    def check(self, condition: bool, detail: str) -> None:
        if not condition:
            raise AssertionError(detail)
        self.count += 1
        print(f"PASS {self.count:02d}: {detail}", flush=True)


class RecordingMapper:
    def __init__(self, response: np.ndarray) -> None:
        self.shape = response.shape
        self.dtype = response.dtype
        self._response = response
        self.requests: list[int] = []

    def __getitem__(self, index: int) -> np.ndarray:
        self.requests.append(int(index))
        return self._response[int(index)]


def _identity(shape: tuple[int, ...]) -> dict[str, object]:
    return {
        "schema": "rift_npz_sealed_protocol_v1",
        "version": 1,
        "data_format": "npz",
        "response_shape": list(shape),
        "response_dtype": "complex64",
        "role_manifest_name": "tiny_adapter_fixture",
        "split_strategy": "fixture",
        "role_ids": {
            "train": [4, 1],
            "validation": [5],
            "reserved_test": [0],
            "unused": [2, 3],
        },
        "response_access": {
            "train_materialized": True,
            "validation_materialized": True,
            "reserved_test_materialized": False,
            "unused_materialized": False,
        },
    }


def _fixture(path: Path, *, compressed: bool) -> np.ndarray:
    shape = (6, 2, 3, 1, 4)
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    response = (0.25 + values + 1j * (0.5 + values)).astype(np.complex64)
    viewpoints = np.arange(shape[0] * 3, dtype=np.float32).reshape(shape[0], 3)
    tx_pos = np.arange(shape[0] * shape[1] * 3, dtype=np.float32).reshape(shape[0], shape[1], 3)
    rx_pos = np.arange(shape[0] * shape[2] * 3, dtype=np.float32).reshape(shape[0], shape[2], 3)
    metadata = {
        "num_chirps_cpi": shape[3],
        "num_adc_samples": shape[4],
        "radar_fc_hz": 10.0e9,
        "radar_bandwidth_hz": 1.0e9,
    }
    writer = np.savez_compressed if compressed else np.savez
    writer(
        path,
        response=response,
        metadata_json=np.asarray(json.dumps(metadata)),
        viewpoint_positions=viewpoints,
        tx_pos=tx_pos,
        rx_pos=rx_pos,
    )
    return response


def _expect_exception(gates: Gates, action, expected: type[BaseException], detail: str) -> None:
    try:
        action()
    except expected:
        gates.check(True, detail)
    else:
        raise AssertionError(detail)


def main() -> None:
    gates = Gates()
    gates.check(
        "torch" not in sys.modules,
        "adapter import and data-free validator require no torch runtime",
    )
    gates.check(
        "file_sha256(" not in Path(adapter.__file__).read_text(encoding="utf-8"),
        "adapter source contains no full-archive SHA-256 call",
    )
    original_preflight = adapter._preflight_b7873200_identity
    original_mapper = adapter._stored_response_memmap
    try:
        # The fixture contains only synthetic six-view data and is kept out of
        # the source tree.  TemporaryDirectory removes it on every normal
        # outcome.
        with tempfile.TemporaryDirectory(prefix="geraf_b7873200_adapter_") as temporary:
            root = Path(temporary)
            stored_path = root / "stored.npz"
            response = _fixture(stored_path, compressed=False)
            with zipfile.ZipFile(stored_path, "r") as archive:
                gates.check(
                    archive.getinfo("response.npy").compress_type == zipfile.ZIP_STORED,
                    "fixture response member is ZIP_STORED without reading its payload",
                )

            events: list[str] = []
            mapper = RecordingMapper(response)

            def fake_preflight(npz_path, manifest_path):
                events.append("preflight")
                gates.check(Path(npz_path) == stored_path and manifest_path == "roles.json",
                            "adapter forwards the archive and manifest to protocol preflight")
                return _identity(response.shape)

            def fake_mapper(path):
                events.append("mapper")
                gates.check(path == str(stored_path.resolve()),
                            "stored mapper is opened only for the resolved archive path")
                return mapper

            adapter._preflight_b7873200_identity = fake_preflight
            adapter._stored_response_memmap = fake_mapper
            arrays, identity = adapter.load_b7873200_sealed_power_arrays(stored_path, "roles.json")

            gates.check(events == ["preflight", "mapper"],
                        "protocol preflight completes before the stored response mapper opens")
            gates.check(
                arrays.path == str(stored_path.resolve())
                and arrays.response.shape == response.shape
                and arrays.response.dtype == response.dtype
                and arrays.num_views == 6
                and arrays.num_tx == 2
                and arrays.num_rx == 3
                and arrays.num_chirps == 1
                and arrays.num_freq == 4,
                "adapter preserves B787PowerArrays path/header and num_* semantics without exposing an ndarray",
            )
            gates.check(
                not hasattr(arrays, "_response_mapper")
                and "_response_mapper" not in vars(arrays),
                "the returned adapter has no mapper attribute that could bypass its role guard",
            )
            gates.check(
                arrays.viewpoint_positions.dtype == np.float64
                and arrays.tx_pos.dtype == np.float64
                and arrays.rx_pos.dtype == np.float64
                and arrays.metadata["num_adc_samples"] == 4,
                "adapter preserves the historical float64 geometry and decoded metadata semantics",
            )
            _expect_exception(
                gates,
                lambda: arrays.response[0],  # type: ignore[index]
                TypeError,
                "header-only response proxy cannot bypass response_view indexing",
            )
            observed_raw = arrays.response_view(4, chirp_average=False)
            observed_mean = arrays.response_view(1)
            gates.check(
                np.array_equal(observed_raw, response[4])
                and np.array_equal(observed_mean, response[1].mean(axis=2))
                and mapper.requests == [4, 1],
                "only requested authorized rows are mapped and chirp averaging matches B787PowerArrays",
            )
            _expect_exception(
                gates,
                lambda: arrays.response_view(0),
                PermissionError,
                "reserved-test response access is denied before the mapper is indexed",
            )
            _expect_exception(
                gates,
                lambda: arrays.response_view(2),
                PermissionError,
                "unused response access is denied before the mapper is indexed",
            )
            _expect_exception(
                gates,
                lambda: setattr(arrays, "_allowed_response_view_indices", frozenset(range(6))),
                FrozenInstanceError,
                "the returned adapter cannot be mutated to widen response-role access",
            )
            gates.check(
                mapper.requests == [4, 1]
                and arrays.allowed_response_view_indices == frozenset({1, 4, 5})
                and arrays.response_payload_materialized is False
                and arrays.response_access_is_restricted is True,
                "denied roles cannot cause raw reads and provenance records the restricted capability",
            )
            gates.check(
                identity == arrays.sealed_protocol_identity
                and identity is not arrays.sealed_protocol_identity,
                "caller receives a copy of the sealed semantic identity",
            )

            compressed_path = root / "compressed.npz"
            compressed_response = _fixture(compressed_path, compressed=True)
            compressed_events: list[str] = []

            def compressed_preflight(_npz_path, _manifest_path):
                compressed_events.append("preflight")
                return _identity(compressed_response.shape)

            def forbidden_mapper(_path):
                compressed_events.append("mapper")
                raise AssertionError("compressed fixture must fail before mapper opening")

            adapter._preflight_b7873200_identity = compressed_preflight
            adapter._stored_response_memmap = forbidden_mapper
            _expect_exception(
                gates,
                lambda: adapter.load_b7873200_sealed_power_arrays(compressed_path, "roles.json"),
                ValueError,
                "compressed response member fails closed rather than using a streaming fallback",
            )
            gates.check(
                compressed_events == ["preflight"],
                "compressed failure occurs after semantic preflight and before any response mapper opens",
            )

            adapter._preflight_b7873200_identity = lambda *_args: {
                **_identity(response.shape),
                "response_access": {"train_materialized": True},
            }
            adapter._stored_response_memmap = fake_mapper
            _expect_exception(
                gates,
                lambda: adapter.load_b7873200_sealed_power_arrays(stored_path, "roles.json"),
                ValueError,
                "an identity that would weaken reserved-role sealing is rejected before mapping",
            )

            overlapping_identity = _identity(response.shape)
            overlapping_identity["role_ids"]["reserved_test"][0] = 4  # type: ignore[index]
            adapter._preflight_b7873200_identity = lambda *_args: overlapping_identity
            _expect_exception(
                gates,
                lambda: adapter.load_b7873200_sealed_power_arrays(stored_path, "roles.json"),
                ValueError,
                "an identity with a reserved/train role overlap is rejected before mapping",
            )
    finally:
        adapter._preflight_b7873200_identity = original_preflight
        adapter._stored_response_memmap = original_mapper
    print(f"GeRaF B7873200 sealed adapter passed: {gates.count} checks.", flush=True)


if __name__ == "__main__":
    main()
