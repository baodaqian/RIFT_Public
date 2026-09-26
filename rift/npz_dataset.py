"""Loading coherent radar NPZs with ground-truth per-view array geometry.

This low-level format loader serves the six-object RIFT dataset as well as
historical sphere/point-target acquisitions. Collection consumers should enter
through rift.rift_dataset.load_object[_contract] to bind object identity and
sealed roles before accessing responses; the eager default here is retained
only for backward compatibility with older callers.

Unlike CSVSimulationDataset (rift/dataset.py), which only carries a per-
viewpoint (dphi, dtheta) pair and relies on forward_operator.get_array_pos's
analytic (theta, phi) -> element-position convention, this npz format was
generated with GROUND-TRUTH absolute Tx/Rx element positions per viewpoint.
See scripts/validate_pec_sphere_coherence.py and project memory
(rift-pec-sphere-coherence-validated) for how this was validated: per-view
specular fits explain ~99.7% of each view's own power, and coherent
multi-view accumulation using this exact geometry behaves correctly (unlike
the old data/AEDT_Sphere_Repeat_CSV, where summing across views collapsed
the signal). Because the positions are exact, get_array_pos is NOT called
for this data source at all -- train.py branches on --data-format npz to use
the tx_pos/rx_pos this module returns directly.

Data format specifics (no frequency array is stored; reverse-engineered):
- `response` [n_view, Tx, Rx, n_chirp, n_adc] complex64: raw FMCW ADC data.
  The n_chirp axis is redundant repeats for this static target (std ~6e-9 vs
  signal ~7e-4 at one channel) -- averaged away here, not Doppler content.
  The n_adc axis behaves like a genuine swept-frequency axis: treating it as
  S(f_i) for f_i = fc - B/2 + i*(B/n_adc) and IFFT-ing over it puts the
  range-profile peak exactly at the geometric near-surface specular range
  for every viewpoint checked -- confirms both the grid direction and the
  sign convention S(f) = exp(-j*2*pi*f/c*R_bistatic) empirically.
- `tx_pos`/`rx_pos` [n_view, Tx/Rx, 3]: absolute element positions in the
  object/world frame, used directly by the forward/range operator's
  arr_pos_tx/arr_pos_rx arguments (they already accept arbitrary positions).
- `viewpoint_positions` [n_view, 3]: absolute array-center position;
  converted to (theta, phi) in radians via the SAME spherical convention
  forward_operator.get_array_pos uses (direction = [sin(theta)cos(phi),
  sin(theta)sin(phi), cos(theta)]) purely so SH-based scenes
  (SHVoxelGridScene/AdaptivePointSHScene) have a viewing angle to evaluate
  their angular basis at -- NOT used to derive array positions.

Per-item tuples deliberately mirror CSVSimulationDataset's 5-tuple
(freqs, dphi, dtheta, magnitude, phase) plus two extra tensors (rx_pos,
tx_pos), so most of train.py's per-viewpoint loop logic (loss, forward
operator call, SH scene evaluation) is unchanged; only the three sites that
call forward_operator.get_array_pos branch on --data-format to use these
extra tensors instead.
"""
import json
import os
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _direction_to_theta_phi(vp):
    """vp: [3] array, absolute radar/array-center position (target at the
    origin). Returns (theta, phi) in radians using the same convention as
    forward_operator.get_array_pos (direction = [sin(theta)cos(phi),
    sin(theta)sin(phi), cos(theta)])."""
    r = float(np.linalg.norm(vp))
    theta = float(np.arccos(np.clip(vp[2] / r, -1.0, 1.0)))
    phi = float(np.arctan2(vp[1], vp[0]))
    return theta, phi


def decode_metadata_json(raw: object) -> dict:
    """Decode scalar or singleton ``metadata_json`` without repr artifacts.

    Older materializers have written this field as a NumPy scalar, a
    singleton array, or UTF-8 bytes.  ``json.loads(str(array))`` only happens
    to work for one of those encodings.  Keeping the decoder here makes the
    NPZ loader independently compatible without changing the historical
    payload format.
    """

    while isinstance(raw, np.ndarray):
        if raw.size != 1:
            raise ValueError(
                "metadata_json must be a scalar or singleton numpy array, "
                f"got shape {raw.shape}"
            )
        raw = raw.reshape(-1)[0]
    if isinstance(raw, np.generic):
        raw = raw.item()
    if isinstance(raw, (bytes, bytearray, np.bytes_)):
        raw = bytes(raw).decode("utf-8")
    if not isinstance(raw, str):
        raise ValueError(f"metadata_json must decode to text, got {type(raw).__name__}")
    try:
        metadata = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("metadata_json is not valid JSON") from exc
    if not isinstance(metadata, dict):
        raise ValueError("metadata_json must encode a JSON object")
    return metadata


def _response_header(path: str) -> Tuple[Tuple[int, ...], np.dtype, bool]:
    """Read the ``response.npy`` header inside an NPZ, not its payload."""

    with zipfile.ZipFile(path) as archive:
        try:
            member = archive.getinfo("response.npy")
        except KeyError as exc:
            raise ValueError(f"{path} is missing required key: response") from exc
        with archive.open(member) as handle:
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(handle)
            elif version in ((2, 0), (3, 0)):
                shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(handle)
            else:
                raise ValueError(f"unsupported response.npy version {version}")
    return tuple(int(value) for value in shape), np.dtype(dtype), bool(fortran_order)


def _discard_bytes(handle, count: int) -> None:
    """Consume a compressed response stream with bounded temporary memory."""

    buffer = bytearray(min(1024 * 1024, max(1, int(count))))
    remaining = int(count)
    while remaining:
        wanted = min(len(buffer), remaining)
        read = handle.readinto(memoryview(buffer)[:wanted])
        if not read:
            raise EOFError(f"response payload ended with {remaining} bytes still required")
        remaining -= int(read)


def _read_exact_bytes(handle, count: int) -> bytes:
    pieces = []
    remaining = int(count)
    while remaining:
        piece = handle.read(min(1024 * 1024, remaining))
        if not piece:
            raise EOFError(f"response payload ended with {remaining} bytes still required")
        pieces.append(piece)
        remaining -= len(piece)
    return b"".join(pieces)


@contextmanager
def _validated_response_stream(
    path: str,
    shape: Tuple[int, ...],
    dtype: np.dtype,
):
    """Open a C-order response member and yield its stream after header checks."""

    archive = zipfile.ZipFile(path)
    handle = archive.open("response.npy")
    try:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            streamed_shape, fortran_order, streamed_dtype = np.lib.format.read_array_header_1_0(handle)
        elif version in ((2, 0), (3, 0)):
            streamed_shape, fortran_order, streamed_dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            raise ValueError(f"unsupported response.npy version {version}")
        if tuple(int(value) for value in streamed_shape) != tuple(shape):
            raise RuntimeError("response.npy header changed while the dataset was being read")
        if np.dtype(streamed_dtype) != np.dtype(dtype):
            raise RuntimeError("response.npy dtype changed while the dataset was being read")
        if fortran_order:
            raise ValueError("lazy response access supports only C-order response.npy arrays")
        yield handle
    finally:
        handle.close()
        archive.close()


def _normalize_view_indices(indices: Iterable[int], num_views: int) -> Tuple[int, ...]:
    """Validate source-view IDs without silently coercing fractional values."""

    normalized = []
    for value in indices:
        integer = int(value)
        if integer != value:
            raise ValueError(f"view index must be integral, got {value!r}")
        if not 0 <= integer < int(num_views):
            raise IndexError(f"view index {integer} is outside [0,{num_views})")
        normalized.append(integer)
    return tuple(normalized)


@dataclass(frozen=True)
class _LazyResponseReader:
    """A bounded-memory, optionally role-restricted response reader.

    ``allowed_view_indices`` is a capability boundary for new sealed-split
    workflows.  It is deliberately unavailable in eager mode: a full response
    array cannot honestly be presented as role restricted after it was read.
    """

    source_path: str
    shape: Tuple[int, ...]
    dtype: np.dtype
    allowed_view_indices: Optional[frozenset] = None

    def _validate_allowed(self, indices: Iterable[int]) -> Tuple[int, ...]:
        normalized = _normalize_view_indices(indices, self.shape[0])
        if self.allowed_view_indices is not None:
            denied = sorted(set(normalized).difference(self.allowed_view_indices))
            if denied:
                raise PermissionError(
                    "response access is restricted to the authorized split; "
                    f"denied source-view IDs: {denied}"
                )
        return normalized

    def restricted_to(self, indices: Iterable[int]) -> "_LazyResponseReader":
        requested = frozenset(self._validate_allowed(indices))
        return _LazyResponseReader(
            source_path=self.source_path,
            shape=self.shape,
            dtype=self.dtype,
            allowed_view_indices=requested,
        )

    def response_view(self, view_index: int) -> np.ndarray:
        return next(self.iter_response_views([view_index]))[1]

    def iter_response_views(self, indices: Iterable[int]):
        requested = self._validate_allowed(indices)
        ordered = sorted(set(requested))
        if not ordered:
            return
        bytes_per_view = int(np.prod(self.shape[1:], dtype=np.int64)) * self.dtype.itemsize
        with _validated_response_stream(self.source_path, self.shape, self.dtype) as handle:
            target_position = 0
            for current_view in range(self.shape[0]):
                selected = (
                    target_position < len(ordered)
                    and current_view == ordered[target_position]
                )
                if selected:
                    raw = _read_exact_bytes(handle, bytes_per_view)
                    yield current_view, np.frombuffer(raw, dtype=self.dtype).reshape(self.shape[1:]).copy()
                    target_position += 1
                    if target_position == len(ordered):
                        return
                else:
                    _discard_bytes(handle, bytes_per_view)


def _validate_npz_contract(
    response_shape: Tuple[int, ...],
    response_dtype: np.dtype,
    response_fortran: bool,
    viewpoint_positions: np.ndarray,
    tx_pos: np.ndarray,
    rx_pos: np.ndarray,
    *,
    lazy: bool,
) -> None:
    if len(response_shape) != 5:
        raise ValueError(
            "response must have shape [view,tx,rx,chirp,freq], "
            f"got {response_shape}"
        )
    if lazy and response_fortran:
        raise ValueError("lazy response access supports only C-order response.npy arrays")
    if not np.issubdtype(response_dtype, np.complexfloating):
        raise ValueError("response must be complex-valued")
    if response_shape[0] != viewpoint_positions.shape[0]:
        raise ValueError("response and viewpoint_positions view counts disagree")
    if tx_pos.shape != (response_shape[0], response_shape[1], 3):
        raise ValueError("tx_pos must have shape [view,tx,3] matching response")
    if rx_pos.shape != (response_shape[0], response_shape[2], 3):
        raise ValueError("rx_pos must have shape [view,rx,3] matching response")


def load_npz_arrays(npz_path, *, load_response=True):
    """Load NPZ metadata/poses and optionally the response payload.

    ``load_response=True`` is the historical default: callers receive the
    full eager ``response`` array and old recipes retain their exact behavior.
    ``load_response=False`` reads only the response header, metadata, and pose
    arrays.  The returned dictionary has ``response is None`` and carries a
    private bounded-memory reader for :func:`get_npz_response_view` and
    :func:`iter_npz_response_views`.
    """

    path = os.fspath(npz_path)
    required = ("metadata_json", "viewpoint_positions", "tx_pos", "rx_pos")
    # ``allow_pickle=True`` preserves compatibility with historical metadata
    # object arrays.  A decoded value must still be a JSON object.
    with np.load(path, allow_pickle=True) as data:
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"{path} is missing required keys: {missing}")
        meta = decode_metadata_json(data["metadata_json"])
        viewpoint_positions = np.asarray(data["viewpoint_positions"])
        tx_pos = np.asarray(data["tx_pos"])
        rx_pos = np.asarray(data["rx_pos"])
        response = np.asarray(data["response"]) if load_response else None

    # Preserve the original eager contract and avoid a second archive pass in
    # legacy recipes.  The old loader returned these five keys immediately
    # after decoding the requested arrays; all header/role checks below belong
    # solely to the new opt-in lazy path.
    if load_response:
        return {
            "response": response,
            "viewpoint_positions": viewpoint_positions,
            "tx_pos": tx_pos,
            "rx_pos": rx_pos,
            "meta": meta,
        }

    response_shape, response_dtype, response_fortran = _response_header(path)
    _validate_npz_contract(
        response_shape,
        response_dtype,
        response_fortran,
        viewpoint_positions,
        tx_pos,
        rx_pos,
        lazy=not load_response,
    )
    reader = _LazyResponseReader(
        source_path=os.path.abspath(path),
        shape=response_shape,
        dtype=response_dtype,
    )
    arrays = {
        "response": None,
        "viewpoint_positions": viewpoint_positions,
        "tx_pos": tx_pos,
        "rx_pos": rx_pos,
        "meta": meta,
    }
    # Do not add keys to the legacy eager return value.  New lazy consumers
    # alone need the response contract and its bounded reader.
    arrays.update({
        "response_shape": response_shape,
        "response_dtype": response_dtype,
        "response_payload_materialized": False,
        "_lazy_response_reader": reader,
    })
    return arrays


def npz_response_num_views(arrays: dict) -> int:
    """Return the response view count without materializing a lazy payload."""

    response = arrays.get("response")
    if response is not None:
        return int(response.shape[0])
    shape = arrays.get("response_shape")
    if shape is None:
        raise RuntimeError("lazy arrays have no response_shape metadata")
    return int(shape[0])


def get_npz_response_view(arrays: dict, view_index: int) -> np.ndarray:
    """Return exactly one source response view.

    For legacy eager dictionaries this is a normal array slice.  In lazy mode
    it streams only the requested view and honours any split restriction.
    """

    response = arrays.get("response")
    if response is not None:
        index = _normalize_view_indices([view_index], response.shape[0])[0]
        return response[index]
    reader = arrays.get("_lazy_response_reader")
    if reader is None:
        raise RuntimeError("lazy response access requires an NPZ response reader")
    return reader.response_view(view_index)


def iter_npz_response_views(arrays: dict, view_indices: Iterable[int]):
    """Yield selected source views once, in ascending ID order.

    The lazy implementation only retains one raw response view at a time.  It
    intentionally does not make arbitrary NPZ roles accessible when the
    dictionary was narrowed with :func:`restrict_npz_response_views`.
    """

    response = arrays.get("response")
    if response is not None:
        ordered = sorted(set(_normalize_view_indices(view_indices, response.shape[0])))
        for index in ordered:
            yield index, response[index]
        return
    reader = arrays.get("_lazy_response_reader")
    if reader is None:
        raise RuntimeError("lazy response access requires an NPZ response reader")
    yield from reader.iter_response_views(view_indices)


def restrict_npz_response_views(arrays: dict, allowed_view_indices: Iterable[int]) -> dict:
    """Return a lazy dictionary that can materialize only specific source IDs.

    This is for new split-aware recipes.  It refuses eager dictionaries because
    an already materialized full response cannot be made sealed retroactively.
    Re-restricting a dictionary may only narrow its existing authorization.
    """

    if arrays.get("response") is not None:
        raise ValueError("role restriction requires load_npz_arrays(..., load_response=False)")
    reader = arrays.get("_lazy_response_reader")
    if reader is None:
        raise RuntimeError("lazy response access requires an NPZ response reader")
    restricted = dict(arrays)
    restricted["_lazy_response_reader"] = reader.restricted_to(allowed_view_indices)
    return restricted


def build_freqs(meta):
    fc = float(meta["radar_fc_hz"])
    bw = float(meta["radar_bandwidth_hz"])
    n_adc = int(meta["num_adc_samples"])
    return (fc - bw / 2) + np.arange(n_adc) * (bw / n_adc)


class PecSphereNPZDataset(Dataset):
    """One item per selected source-view ID.

    Legacy callers hand in an eager ``response`` array.  New callers may pass
    a lazy, role-restricted dictionary returned by
    :func:`restrict_npz_response_views`; in that case this constructor streams
    only its selected authorized views and never retains a raw full payload.
    The finished tensor items remain eagerly cached exactly as in the original
    dataset implementation.
    """

    def __init__(self, arrays, indices, source_path=None):
        meta = arrays["meta"]
        self.sphere_radius = float(meta.get("target_radius_m", 1.0))
        freqs_np = build_freqs(meta)
        self.freqs = torch.tensor(freqs_np, dtype=torch.float32)
        # Preserve canonical source-view IDs for exact adaptive-resume
        # provenance.  They do not alter item ordering or data loading.
        self.indices = np.asarray(indices, dtype=np.int64).copy()
        self.source_path = "" if source_path is None else str(source_path)

        vp = arrays["viewpoint_positions"]
        tx_pos = arrays["tx_pos"]
        rx_pos = arrays["rx_pos"]

        def make_item(v, response_view):
            cube = response_view.mean(axis=2)  # [Tx, Rx, n_adc] complex64, average redundant chirps
            num_tx, num_rx, n_adc = cube.shape
            # Tx-outer/Rx-inner flat channel order, matching train.py's
            # reshape_measured_cubes docstring/convention for CSV data.
            flat = cube.reshape(num_tx * num_rx, n_adc).T  # [n_adc, Tx*Rx]
            magnitude = torch.tensor(np.abs(flat), dtype=torch.float32)
            phase = torch.tensor(np.angle(flat), dtype=torch.float32)
            theta, phi = _direction_to_theta_phi(vp[v])
            return (
                torch.tensor([theta], dtype=torch.float32),
                torch.tensor([phi], dtype=torch.float32),
                magnitude,
                phase,
                torch.tensor(rx_pos[v], dtype=torch.float32),
                torch.tensor(tx_pos[v], dtype=torch.float32),
            )

        # ``iter_npz_response_views`` reads source IDs in ascending order for
        # one-pass compressed-NPZ streaming.  Store finished items by source
        # ID, then restore the caller's original (typically seeded-permuted)
        # order so optimization trajectories remain unchanged.
        repeat_count = {}
        for view in self.indices:
            view = int(view)
            repeat_count[view] = repeat_count.get(view, 0) + 1
        item_by_view = {}
        for view, response_view in iter_npz_response_views(arrays, self.indices):
            view = int(view)
            # Direct callers can provide duplicate IDs.  Preserve the old
            # behavior of creating separate tensor items for every occurrence
            # without retaining the raw response view after this iteration.
            item_by_view[view] = [
                make_item(view, response_view)
                for _ in range(repeat_count[view])
            ]
        self.items = [item_by_view[int(view)].pop(0) for view in self.indices]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        dtheta, dphi, magnitude, phase, rx_pos, tx_pos = self.items[idx]
        # dtheta/dphi are length-1 tensors: forward_operator.get_array_pos
        # and SHVoxelGridScene/AdaptivePointSHScene.active_scatterers only
        # ever read element [0] after squeeze(0), the same as
        # CSVSimulationDataset's per-frequency-row (but constant) columns.
        return self.freqs, dphi, dtheta, magnitude, phase, rx_pos, tx_pos


def polar_cap_split(viewpoint_positions, num_val, cap_axis, num_train, num_test, rng):
    """Held-out set = the ``num_val`` views whose look direction is closest to
    ``cap_axis``, i.e. a contiguous polar CAP rather than a scattered subset.

    Why this exists: with a random (or tail) split the held-out views sit
    interspersed among the training views, so predicting them is INTERPOLATION on
    the view sphere -- the same task SH-SAS performs, which is why the
    "extrapolation, not densification" axis was retracted 2026-08-05. A cap split
    puts every held-out direction OUTSIDE the convex hull of the training
    directions, so the model must predict an aspect it has never seen. A
    band-limited angular interpolator cannot do this (measured: 28.7% -> 62.2% on
    a power-matched cap, `scripts/eval_nvs_view_ladder.py --mode cap`), whereas
    the forward operator renders an out-of-hull viewpoint with exactly the same
    computation it uses in-hull.

    The training views are returned in a SEEDED PERMUTATION of the non-cap set,
    never in cap-distance order: Fibonacci view indices are elevation-ordered, so
    an ordered training stream would feed the optimizer a slow sweep across the
    sphere instead of an i.i.d. one.
    """
    positions = np.asarray(viewpoint_positions, dtype=np.float64)
    unit = positions / np.linalg.norm(positions, axis=1, keepdims=True)
    axis = np.asarray(cap_axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)

    projection = unit @ axis
    order = np.argsort(-projection)                    # most cap-aligned first
    val_idx = order[:num_val]
    remainder = rng.permutation(order[num_val:])       # de-order before slicing
    test_idx = remainder[:num_test]
    train_idx = remainder[num_test:num_test + num_train]

    half_angle = float(np.degrees(np.arccos(np.clip(projection[val_idx].min(), -1.0, 1.0))))
    return train_idx, val_idx, test_idx, half_angle


def build_npz_dataloaders(npz_path, num_train, num_val, num_test, seed=None,
                          val_from_tail=False, cap_axis=None, *,
                          lazy_response=False, include_test=True):
    """Mirrors train.py's build_dataloaders: disjoint random viewpoint
    splits, batch_size=1, no shuffling beyond the split itself (matching
    CSVSimulationDataset's DataLoader construction).

    val_from_tail=False (default, unchanged): val = perm[num_train:...], so the
    held-out set moves when num_train changes. val_from_tail=True: reserve the
    LAST num_val (+num_test) indices of the seed-fixed permutation as a FIXED
    held-out set and grow train from the front -- required for a num_train sweep
    (Nyquist-transition ablation) so every density level is scored on the same
    interpolation targets. Held-out views stay randomly interleaved over the
    view-manifold because perm is a random permutation.

    cap_axis=(x,y,z) switches to an angular-EXTRAPOLATION split (see
    ``polar_cap_split``) and takes precedence over val_from_tail. This is the
    only split where held-out views lie outside the training hull; every other
    mode interpolates.

    ``lazy_response=False`` and ``include_test=True`` are the historical
    behavior: the whole payload and all three datasets are materialized.
    New sealed-split recipes can opt into
    ``lazy_response=True, include_test=False``.  That path reads only NPZ
    metadata/poses before splitting, authorizes only train+validation source
    IDs, and returns ``None`` as the third loader.  Test IDs are still
    calculated for provenance, but their response payload is never
    materialized or exposed through either returned dataset."""
    from torch.utils.data import DataLoader

    arrays = load_npz_arrays(npz_path, load_response=not lazy_response)
    n_view = npz_response_num_views(arrays)
    n_wanted = num_train + num_val + num_test
    if n_wanted > n_view:
        raise ValueError(f"Requested {n_wanted} total viewpoints but {npz_path} only has {n_view}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_view)
    if cap_axis is not None:
        train_idx, val_idx, test_idx, half_angle = polar_cap_split(
            arrays["viewpoint_positions"], num_val, cap_axis, num_train, num_test, rng)
        print(f"Polar-cap (EXTRAPOLATION) split about axis {tuple(float(a) for a in cap_axis)}: "
              f"{len(val_idx)} held-out views inside a {half_angle:.1f} deg cap, "
              f"{len(train_idx)} training views drawn from outside it. Held-out directions lie "
              f"OUTSIDE the convex hull of the training directions.")
    elif val_from_tail:
        # Fixed tail held-out set; train grows from the front and must not reach it.
        val_idx = perm[n_view - num_val:]
        test_idx = perm[n_view - num_val - num_test:n_view - num_val]
        train_idx = perm[:num_train]
    else:
        train_idx = perm[:num_train]
        val_idx = perm[num_train:num_train + num_val]
        test_idx = perm[num_train + num_val:num_train + num_val + num_test]

    if lazy_response:
        permitted = [train_idx, val_idx]
        if include_test:
            permitted.append(test_idx)
        arrays = restrict_npz_response_views(arrays, np.concatenate(permitted))

    training_dataset = PecSphereNPZDataset(arrays, train_idx, source_path=npz_path)
    validation_dataset = PecSphereNPZDataset(arrays, val_idx, source_path=npz_path)
    test_dataset = (
        PecSphereNPZDataset(arrays, test_idx, source_path=npz_path)
        if include_test
        else None
    )

    training_data_loader = DataLoader(training_dataset, batch_size=1)
    validation_data_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
    test_data_loader = (
        DataLoader(test_dataset, batch_size=1, shuffle=False)
        if test_dataset is not None
        else None
    )
    return training_data_loader, validation_data_loader, test_data_loader
