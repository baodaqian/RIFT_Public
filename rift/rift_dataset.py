"""Portable six-object RIFT dataset catalog and object-bound sealed splits.

An object is a separate radar scene, never an extra view of another object.
The protocol extends the existing 3200/1000/1000 split with explicit object
identity: identical acquisition poses alone cannot identify a scene.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping
import zipfile

import numpy as np

from .antenna_selection import (selection, validate_selection, acquisition_label, select_arrays)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "protocols" / "rift_dataset.json"
DEFAULT_ROOT = PROJECT_ROOT / "data" / "RIFT_dataset"
DATASET_ID = "rift_dataset_v1"
PARENT_NUM_TRAIN = 3200
DEFAULT_NUM_TRAIN = 2400


def training_count(num_train=PARENT_NUM_TRAIN) -> int:
    if (isinstance(num_train, (bool, np.bool_))
            or not isinstance(num_train, (int, np.integer))
            or not 1 <= num_train <= PARENT_NUM_TRAIN):
        raise ValueError("RIFT num_train must be an integer in [1, 3200]")
    return int(num_train)


def training_selection(num_train) -> dict:
    num_train = training_count(num_train)
    ids = role_ids(num_train)["train"]
    return {"schema": "rift_nested_training_subset_v1", "num_train": num_train,
            "parent_num_train": PARENT_NUM_TRAIN, "seed": 42,
            "selection": "prefix_of_parent_pcg64_permutation",
            "train_ids_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()}


def catalog() -> dict:
    return json.loads(CATALOG_PATH.read_text())


def object_spec(name: str) -> dict:
    for spec in catalog()["objects"]:
        if name in (spec["object_id"], *spec["aliases"]):
            return spec
    raise ValueError(f"Unknown RIFT dataset object: {name!r}")


def object_identity(name: str) -> dict:
    return {"dataset_id": DATASET_ID, "object_id": object_spec(name)["object_id"]}


def metadata_object_id(meta: Mapping) -> str:
    if str(meta.get("target_type", "")).lower() == "b787":
        return "b787"
    if meta.get("target_type") == "mesh":
        return object_spec(str(meta.get("target_id")))["object_id"]
    raise ValueError("Expected a registered RIFT dataset mesh or B787 archive")


def manifest_name(name: str, num_train=PARENT_NUM_TRAIN, antenna_selection=None) -> str:
    num_train = training_count(num_train)
    return (f"rift_dataset_{object_spec(name)['object_id']}_seed42_train{num_train}_val1000_test1000_v1"
            + ("_" + acquisition_label(antenna_selection) if antenna_selection else ""))


def role_ids(num_train=PARENT_NUM_TRAIN) -> dict:
    num_train = training_count(num_train)
    permutation = np.random.Generator(np.random.PCG64(42)).permutation(10000)
    return {"train": permutation[:num_train].tolist(),
            "validation": permutation[9000:].tolist(),
            "reserved_test": permutation[8000:9000].tolist(),
            "unused": permutation[num_train:8000].tolist()}


def role_manifest(name: str, num_train=PARENT_NUM_TRAIN, antenna_selection=None) -> dict:
    num_train = training_count(num_train)
    roles = role_ids(num_train)
    return {"schema_version": 1, "name": manifest_name(name, num_train, antenna_selection),
            **({"antenna_selection": validate_selection(antenna_selection)} if antenna_selection else {}),
            **({"training_selection": training_selection(num_train)} if num_train != PARENT_NUM_TRAIN else {}),
            "dataset": {**object_identity(name), "num_views": 10000,
                        "response_shape": [10000, 16, 16, 1, 600], "response_dtype": "complex64"},
            "split": {"strategy": "fixed_tail_subsampled", "seed": 42,
                      "complete_partition": True, "test_sealed": True, "unused_sealed": True,
                      "num_train": num_train, "num_validation": 1000, "num_test": 1000,
                      "num_unused": 8000-num_train, "train_indices": roles["train"],
                      "validation_indices": roles["validation"],
                      "test_indices": roles["reserved_test"], "unused_indices": roles["unused"]}}


def validate_manifest_object(manifest: Mapping, metadata: Mapping) -> dict | None:
    """Bind an opt-in collection manifest to actual metadata before payload reads."""
    if not isinstance(manifest, Mapping):
        raise ValueError("Role manifest must be a JSON object")
    declared = manifest.get("dataset", {})
    if not isinstance(declared, Mapping):
        raise ValueError("Role manifest dataset must be an object")
    if declared.get("dataset_id") != DATASET_ID:
        if str(manifest.get("name", "")).startswith("rift_dataset_"):
            raise ValueError("RIFT dataset manifest is missing its dataset identity")
        return None  # Preserve historical single-object contracts byte-for-byte.
    expected = object_identity(declared.get("object_id"))
    if expected["object_id"] != declared.get("object_id"):
        raise ValueError("Manifest must use the canonical object ID")
    if metadata_object_id(metadata) != expected["object_id"]:
        raise ValueError("RIFT dataset manifest object does not match the NPZ metadata")
    split = manifest.get("split")
    if not isinstance(split, Mapping):
        raise ValueError("RIFT dataset manifest split must be an object")
    num_train = training_count(split.get("num_train"))
    registered = role_manifest(expected["object_id"], num_train, validate_selection(manifest.get("antenna_selection")))
    if manifest.get("name") != registered["name"]:
        raise ValueError("RIFT dataset role manifest name does not match its object")
    if (not _exact_value(manifest.get("split"), registered["split"])
            or not _exact_value(manifest.get("training_selection"), registered.get("training_selection"))):
        raise ValueError("RIFT dataset requires the registered PCG64(seed=42) split")
    validate_metadata(metadata)
    return expected


def validate_metadata(meta: Mapping) -> None:
    metadata_object_id(meta)
    if meta.get("experiment") != "sphere10k":
        raise ValueError("RIFT dataset requires sphere10k acquisition metadata")
    if meta.get("viewpoint_sampling") != "fibonacci_sphere":
        raise ValueError("RIFT dataset requires Fibonacci sphere viewpoints")
    if not np.array_equal(np.asarray(meta.get("target_position_m")), np.zeros(3)):
        raise ValueError("RIFT dataset targets must remain centered at the origin")
    for key, expected in (("radar_fc_hz", 1e10), ("radar_bandwidth_hz", 3e9),
                          ("num_adc_samples", 600), ("num_chirps_cpi", 1)):
        if float(meta.get(key, -1)) != expected:
            raise ValueError(f"RIFT dataset requires {key}={expected}")
    if not np.isclose(float(meta.get("scaled_max_extent_m", -1)), 0.1, rtol=1e-6):
        raise ValueError("RIFT dataset meshes must retain the simulator's 0.1 m scale")


def collection_manifest(path: str | Path) -> bool:
    """Recognize explicit opt-in, not an arbitrary alternate NPZ path."""
    try:
        manifest = json.loads(Path(path).read_text())
    except FileNotFoundError:
        return False
    if not isinstance(manifest, Mapping) or not isinstance(manifest.get("dataset", {}), Mapping):
        raise ValueError("Role manifest and its dataset must be JSON objects")
    return manifest.get("dataset", {}).get("dataset_id") == DATASET_ID


def _exact_value(observed, expected) -> bool:
    """JSON semantics without Python's True == 1 or 1.0 == 1 coercions."""
    if isinstance(expected, dict):
        return (isinstance(observed, Mapping) and set(observed) == set(expected)
                and all(_exact_value(observed[key], value) for key, value in expected.items()))
    if isinstance(expected, list):
        return (isinstance(observed, list) and len(observed) == len(expected)
                and all(_exact_value(left, right) for left, right in zip(observed, expected)))
    if isinstance(expected, bool):
        return type(observed) is bool and observed == expected
    if isinstance(expected, int):
        return (isinstance(observed, (int, np.integer)) and not isinstance(observed, (bool, np.bool_))
                and observed == expected)
    return type(observed) is type(expected) and observed == expected


def _object_contract(name: str, num_train=PARENT_NUM_TRAIN, antenna_selection=None, source_geometry_sha256=None) -> dict:
    num_train = training_count(num_train)
    acquisition = validate_selection(antenna_selection)
    extra = {}
    shape = [10000, 16, 16, 1, 600]
    if acquisition is not None:
        if (not isinstance(source_geometry_sha256, str) or len(source_geometry_sha256) != 64
                or any(c not in "0123456789abcdef" for c in source_geometry_sha256)):
            raise ValueError("Selected acquisition requires a source geometry SHA-256")
        extra = dict(antenna_selection=acquisition, source_geometry_sha256=source_geometry_sha256,
                     source_response_shape=shape.copy())
        shape[1:3] = [acquisition["num_tx"], acquisition["num_rx"]]
    return {**extra, "schema": "rift_npz_sealed_protocol_v1", "version": 1, "data_format": "npz",
            "response_shape": shape, "response_dtype": "complex64",
            "role_manifest_name": manifest_name(name, num_train, acquisition),
            "split_strategy": "fixed_tail_subsampled", "role_ids": role_ids(num_train),
            **({"training_selection": training_selection(num_train)} if num_train != PARENT_NUM_TRAIN else {}),
            "dataset_identity": object_identity(name),
            "response_access": {"train_materialized": True, "validation_materialized": True,
                                "reserved_test_materialized": False, "unused_materialized": False}}


def collection_contract(contract: Mapping) -> dict | None:
    """Shared baseline contract, or None for a historical B787-only contract."""
    if not isinstance(contract, Mapping):
        raise ValueError("Sealed protocol contract must be a mapping")
    identity = contract.get("dataset_identity")
    if identity is None:
        if str(contract.get("role_manifest_name", "")).startswith("rift_dataset_"):
            raise ValueError("RIFT dataset contract is missing its object identity")
        return None
    if not isinstance(identity, Mapping) or identity != object_identity(identity.get("object_id")):
        raise ValueError("Invalid RIFT dataset object identity")
    expected = _object_contract(identity["object_id"], len(contract.get("role_ids", {}).get("train", ())),
                                contract.get("antenna_selection"), contract.get("source_geometry_sha256"))
    if contract.get("training_selection") != expected.get("training_selection"):
        raise ValueError("RIFT dataset contract changed the registered training selection")
    if any(not _exact_value(contract.get(key), value) for key, value in expected.items()):
        raise ValueError("RIFT dataset contract changed the registered acquisition or sealed roles")
    return expected


def object_paths(root: str | Path, name: str) -> tuple[Path, Path]:
    name = object_spec(name)["object_id"]
    root = Path(root).absolute()
    return root / "objects" / f"{name}.npz", root / "splits" / f"{name}.json"


def resolve_object_inputs(*, object_name=None, dataset_root=DEFAULT_ROOT,
                          npz_path=None, role_manifest_path=None) -> tuple[Path, Path]:
    """Resolve named objects or explicit paths; never silently override a conflict.

    CLI callers should pass None for unprovided legacy defaults, then apply those
    defaults only when no object was selected. Paths may be symlink aliases of
    the same files, so the original standalone B787 archive remains usable.
    """
    if object_name is None:
        if npz_path is None or role_manifest_path is None:
            raise ValueError("Select --object or supply both NPZ and role-manifest paths")
        return Path(npz_path).absolute(), Path(role_manifest_path).absolute()
    selected = object_paths(dataset_root, object_name)
    for explicit, expected, label in zip((npz_path, role_manifest_path), selected, ("NPZ", "role manifest")):
        if explicit is not None and Path(explicit).resolve() != expected.resolve():
            if label == "role manifest" and Path(explicit).is_file():
                declared = json.loads(Path(explicit).read_text()).get("dataset", {})
                if (declared.get("dataset_id") == DATASET_ID
                        and declared.get("object_id") == object_spec(object_name)["object_id"]):
                    selected = (selected[0], Path(explicit).absolute())
                    continue  # Full split/header validation remains in the public loader.
            raise ValueError(f"Conflicting {label} path for RIFT dataset object {object_name!r}")
    return selected


def evaluation_role_indices(contract: Mapping, role: str, *, allow_reserved_test=False) -> np.ndarray:
    """Choose an explicit role; test is opt-in and unused is never an eval role."""
    identity = collection_contract(contract)
    if identity is None:
        raise ValueError("A RIFT dataset object-bound contract is required")
    role = {"val": "validation", "test": "reserved_test"}.get(role, role)
    if role == "reserved_test" and allow_reserved_test is not True:
        raise PermissionError("Reserved-test responses require explicit allow_reserved_test=True")
    if role not in ("train", "validation", "reserved_test"):
        raise ValueError(f"Unsupported evaluation role {role!r}; unused responses remain sealed")
    return np.asarray(identity["role_ids"][role], dtype=np.int64)


def load_object_contract(npz_path, role_manifest_path, *,
                         response_roles=("train", "validation"),
                         allow_reserved_test=False, num_train=None, num_tx=None, num_rx=None,
                         tx_indices=None, rx_indices=None) -> tuple[dict, dict]:
    """Metadata-only public ingress for the collection, independent of train.py.

    The arrays contain response=None and a lazy capability narrowed to the
    requested roles. The returned experiment contract is unchanged; its legacy
    response_access fields describe the training policy, not this handle's reads.
    """
    from .npz_dataset import load_npz_arrays, restrict_npz_response_views

    manifest_path = Path(role_manifest_path).absolute()
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, Mapping) or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("RIFT dataset role manifest requires schema_version=1")
    arrays = load_npz_arrays(npz_path, load_response=False)
    identity = validate_manifest_object(manifest, arrays["meta"])
    if identity is None:
        raise ValueError("A RIFT dataset object-bound manifest is required")
    manifest_count = manifest["split"]["num_train"]
    if num_train is not None:
        num_train = training_count(num_train)
        if num_train > manifest_count:
            raise ValueError("Cannot expand a sealed manifest's training role")
    else:
        num_train = manifest_count
    expected_dataset = role_manifest(identity["object_id"])["dataset"]
    if any(not _exact_value(manifest["dataset"].get(key), value) for key, value in expected_dataset.items()):
        raise ValueError("RIFT dataset manifest acquisition/header changed")
    if (arrays.get("response") is not None
            or tuple(arrays.get("response_shape", ())) != (10000, 16, 16, 1, 600)
            or np.dtype(arrays.get("response_dtype")) != np.dtype("complex64")):
        raise ValueError("RIFT dataset requires a lazy [10000,16,16,1,600] complex64 response")
    for key, shape in (("viewpoint_positions", (10000, 3)),
                       ("tx_pos", (10000, 16, 3)), ("rx_pos", (10000, 16, 3))):
        value = np.asarray(arrays.get(key))
        if value.shape != shape or value.dtype != np.float64 or not np.isfinite(value).all():
            raise ValueError(f"RIFT dataset {key} must be finite float64 with shape {shape}")
    if not np.allclose(np.linalg.norm(arrays["viewpoint_positions"], axis=1), 10.0, rtol=0, atol=1e-8):
        raise ValueError("RIFT dataset viewpoints must lie on the 10 m sphere")
    with zipfile.ZipFile(npz_path) as archive:
        if archive.getinfo("response.npy").compress_type != zipfile.ZIP_STORED:
            raise ValueError("RIFT dataset requires an uncompressed, directly seekable response")
    acquisition = validate_selection(manifest.get("antenna_selection"))
    if any(v is not None for v in (num_tx, num_rx, tx_indices, rx_indices)):
        if acquisition is not None:
            if ((num_tx is not None and num_tx != acquisition['num_tx'])
                    or (num_rx is not None and num_rx != acquisition['num_rx'])):
                raise ValueError("Requested antennas conflict with the sealed manifest")
            tx_indices = acquisition['tx_indices'] if tx_indices is None else tx_indices
            rx_indices = acquisition['rx_indices'] if rx_indices is None else rx_indices
        requested = selection(num_tx, num_rx, tx_indices, rx_indices)
        if acquisition is not None and requested != acquisition:
            raise ValueError("Requested antennas conflict with the sealed manifest")
        acquisition = requested
    arrays = select_arrays(arrays, acquisition)
    contract = {**_object_contract(identity["object_id"], num_train, acquisition,
                                   arrays.get("source_geometry_sha256")),
                "source_path": str(Path(npz_path).absolute()),
                "role_manifest_path": str(manifest_path)}
    if isinstance(response_roles, str) or not response_roles:
        raise ValueError("response_roles must be a nonempty sequence of named roles")
    selected = [evaluation_role_indices(contract, role, allow_reserved_test=allow_reserved_test)
                for role in response_roles]
    return restrict_npz_response_views(arrays, np.concatenate(selected)), contract


def load_object(name, dataset_root=DEFAULT_ROOT, **response_options) -> tuple[dict, dict]:
    """Load one named object/alias, never pool independent scattering scenes."""
    arrays, contract = load_object_contract(*object_paths(dataset_root, name), **response_options)
    if contract["dataset_identity"] != object_identity(name):
        raise ValueError("RIFT dataset object files do not match the selected object")
    return arrays, contract


def validate_checkpoint_object(checkpoint_or_cache: Mapping, expected_contract: Mapping) -> dict:
    """Reject absent/conflicting object identities without inspecting tensor state.

    This is an object-identity gate, not a replacement for method-specific role,
    optimizer, recipe, pose, or normalization validation. Do not infer identity
    from a filename, identical poses, or command-line args on legacy checkpoints.
    """
    expected = collection_contract(expected_contract)
    if expected is None or not isinstance(checkpoint_or_cache, Mapping):
        raise ValueError("Checkpoint/cache and expected RIFT dataset contract must be mappings")
    containers = {"sealed_npz_protocol_contract", "sealed_protocol_contract", "sealed_protocol_identity",
                  "run_identity", "cache_recipe", "recipe", "contract", "stage1_record", "stage1_recipe",
                  "sugavanam_ertin_b7873200_stage1", "generic_final_state", "provenance"}
    pending, found, seen = [checkpoint_or_cache], [], set()
    while pending:
        record = pending.pop()
        if id(record) in seen:
            continue
        seen.add(id(record))
        if "dataset_identity" in record:
            found.append(record["dataset_identity"])
        if "dataset_id" in record and "object_id" in record:
            found.append({key: record[key] for key in ("dataset_id", "object_id")})
        for key in containers:
            if isinstance(record.get(key), Mapping):
                pending.append(record[key])
    if not found:
        raise ValueError("Checkpoint/cache lacks RIFT dataset object identity; legacy data cannot be relabeled")
    if any(not _exact_value(value, expected["dataset_identity"]) for value in found):
        raise ValueError("Checkpoint/cache object identity does not match the selected RIFT dataset object")
    return dict(expected["dataset_identity"])


def geometry_transform(name, metadata: Mapping) -> dict:
    """Explicit evaluation frame from the catalog, never guessed from mesh size/name."""
    scale = float(metadata.get("scale_factor", float("nan")))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("RIFT dataset geometry requires its positive simulator scale_factor")
    frame = object_spec(name)["geometry_input_frame"]
    if frame != "source_model_units":
        raise ValueError(f"Unknown registered mesh input frame: {frame!r}")
    return {"centering": "source_aabb_center", "scale_factor": scale,
            "units": "metres", "input_frame": frame}


def geometry_reference(name, dataset_root=DEFAULT_ROOT, *, mesh_path=None) -> dict:
    """Locate truth for evaluation only; missing geometry has no B787 fallback."""
    arrays, contract = load_object(name, dataset_root)
    spec = object_spec(name)
    root = Path(dataset_root).absolute()
    if mesh_path is None:
        assembled = json.loads((root / "dataset_manifest.json").read_text())
        rows = [row for row in assembled.get("objects", []) if row.get("object_id") == object_spec(name)["object_id"]]
        if len(rows) != 1:
            raise FileNotFoundError(f"No local registered mesh for {name}; supply its original --stl/mesh path explicitly")
        if rows[0].get("geometry"):
            mesh_path = root / rows[0]["geometry"]
        else:
            # Restored originals may arrive after metadata-only assembly. Use the
            # NPZ's exact basename, or the registered name when absent (B787).
            # Never guess another object's geometry.
            filename = arrays["meta"].get("source_stl_filename", spec["geometry_filename"])
            if (not isinstance(filename, str) or not filename or filename in (".", "..")
                    or "/" in filename or "\\" in filename
                    or not (root / "meshes" / filename).is_file()):
                raise FileNotFoundError(f"No local registered mesh for {name}; supply its original --stl/mesh path explicitly")
            mesh_path = root / "meshes" / filename
    mesh_path = Path(mesh_path).absolute()
    if not mesh_path.is_file():
        raise FileNotFoundError(f"RIFT dataset geometry is unavailable: {mesh_path}")
    meta = arrays["meta"]
    return {"mesh_path": mesh_path, "metadata": dict(meta),
            "dataset_identity": contract["dataset_identity"],
            "transform": geometry_transform(name, meta)}


def transform_mesh_vertices(vertices, metadata: Mapping, *, input_frame="source_model_units") -> np.ndarray:
    """Center/scale an original STL in memory; all six objects use this protocol."""
    if input_frame != "source_model_units":
        raise ValueError(f"Unknown mesh input frame: {input_frame!r}; expected original model units")
    values = np.asarray(vertices, dtype=np.float64)
    if values.ndim < 2 or values.shape[-1] != 3 or not values.size or not np.isfinite(values).all():
        raise ValueError("Mesh vertices must be finite (...,3) coordinates")
    flat = values.reshape(-1, 3)
    lower, upper = flat.min(axis=0), flat.max(axis=0)
    dimensions = np.asarray(metadata.get("raw_dimensions_model_units"), dtype=np.float64)
    scale = float(metadata.get("scale_factor", float("nan")))
    if (dimensions.shape != (3,) or not np.isfinite(dimensions).all() or np.any(dimensions <= 0)
            or not np.allclose(upper - lower, dimensions, rtol=1e-5, atol=1e-6)):
        raise ValueError("Mesh dimensions do not match the selected object's source metadata")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Mesh metadata needs a finite positive simulator scale_factor")
    scaled = np.asarray(metadata.get("scaled_dimensions_m"), dtype=np.float64)
    if scaled.shape != (3,) or not np.allclose(dimensions * scale, scaled, rtol=1e-5, atol=1e-8):
        raise ValueError("Mesh simulator scaling disagrees with its recorded metric dimensions")
    return (values - (lower + upper) / 2) * scale
