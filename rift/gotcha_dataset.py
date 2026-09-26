"""Lazy, role-restricted ingress for the combined native GOTCHA acquisition.

The new 250/55/55 sector split is intentionally distinct from both the old
288/36/36 experiment and the all-training transfer run. A viewpoint here is a
pass/sector, containing every native pulse. Polarizations are response channels,
not extra viewpoints. No response member is read during planning.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct
from typing import Mapping
import zipfile

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path('/storage/home/hcoda1/1/dbao31/scratch/GOTCHA-CP_Combined')
PROTOCOL_PATH = PROJECT_ROOT / 'protocols/gotcha_dataset.json'
POLARIZATIONS = ('hh', 'hv', 'vh', 'vv')
SCHEMA = 'rift_gotcha_training_contract_v1'
SOURCE_SCHEMA = 'rift_gotcha_all_data_native_shard_v2'
C = 299792458.0
DEFAULT_NUM_TRAIN = 1500


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def catalog() -> dict:
    return json.loads(PROTOCOL_PATH.read_text())


@dataclass(frozen=True)
class Region:
    name: str
    target_id: str
    translation_m: tuple
    rotation_local_to_native: tuple
    half_extent_m: float
    placement_provenance: str

    def __post_init__(self):
        rotation = np.asarray(self.rotation_local_to_native, dtype=np.float64)
        center = np.asarray(self.translation_m, dtype=np.float64)
        if (not self.name or not self.target_id or not self.placement_provenance
                or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in self.name)):
            raise ValueError('Region requires a safe name, target identity and placement provenance')
        if center.shape != (3,) or rotation.shape != (3, 3) or not np.isfinite(center).all() or not np.isfinite(rotation).all():
            raise ValueError('Region needs finite translation[3] and rotation[3,3]')
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-12, rtol=0) or not np.isclose(np.linalg.det(rotation), 1, atol=2e-12, rtol=0):
            raise ValueError('Region rotation must be proper and orthonormal')
        if not np.isfinite(self.half_extent_m) or self.half_extent_m <= 0:
            raise ValueError('Region half extent must be positive and finite')

    def as_dict(self):
        return dict(name=self.name, target_id=self.target_id,
                    translation_m=list(self.translation_m),
                    rotation_local_to_native=[list(row) for row in self.rotation_local_to_native],
                    half_extent_m=float(self.half_extent_m),
                    placement_provenance=self.placement_provenance)

    def to_local(self, native_xyz):
        return (np.asarray(native_xyz, dtype=np.float64) - self.translation_m) @ np.asarray(self.rotation_local_to_native)


def load_region(name='camry', config_path=None) -> Region:
    regions = catalog()['regions']
    if config_path is not None:
        extra = json.loads(Path(config_path).read_text())
        if extra.get('schema') != 'rift_gotcha_regions_v1' or not isinstance(extra.get('regions'), dict):
            raise ValueError('Region config needs schema rift_gotcha_regions_v1 and a regions mapping')
        if set(extra['regions']) & set(regions):
            raise ValueError('Custom regions must use new names; registered placements cannot be replaced')
        regions.update(extra['regions'])
    if name not in regions:
        raise ValueError(f'Unknown region {name!r}; available: {sorted(regions)}')
    value = regions[name]
    return Region(name=name, target_id=value['target_id'],
                  translation_m=tuple(value['translation_m']),
                  rotation_local_to_native=tuple(tuple(r) for r in value['rotation_local_to_native']),
                  half_extent_m=float(value['half_extent_m']),
                  placement_provenance=value['placement_provenance'])


def sector_split(seed=42) -> dict[str, tuple[int, ...]]:
    if isinstance(seed, bool) or int(seed) != seed or seed != 42:
        raise ValueError('The registered GOTCHA split fixes seed 42')
    ids = np.random.Generator(np.random.PCG64(seed)).permutation(np.arange(1, 361))
    return {role: tuple(sorted(map(int, values))) for role, values in
            zip(('train', 'validation', 'test'), (ids[:250], ids[250:305], ids[305:]))}


def selection(passes=range(1, 9), polarizations=('hh',)):
    passes = tuple(passes)
    pols = tuple(str(p).lower() for p in polarizations)
    if not passes or any(type(p) is not int or p not in range(1, 9) for p in passes) or len(set(passes)) != len(passes):
        raise ValueError('Passes must be distinct integers in 1..8')
    if not pols or any(p not in POLARIZATIONS for p in pols) or len(set(pols)) != len(pols):
        raise ValueError('Polarizations must be distinct HH/HV/VH/VV channels')
    return tuple(sorted(passes)), tuple(p for p in POLARIZATIONS if p in pols)


def training_sectors(passes=range(1, 9), num_train=None):
    """Nested, balanced prefixes in permutation order, retaining native pulses.

    Remainders use a separate seed-42 pass permutation. For 1500/8 this
    selects 187 sectors/pass and the next sector in passes 3, 4, 5 and 8.
    None preserves the frozen parent contract for low-level/legacy callers.
    """
    passes, _ = selection(passes)
    maximum = 250 * len(passes)
    num_train = maximum if num_train is None else num_train
    if (isinstance(num_train, (bool, np.bool_))
            or not isinstance(num_train, (int, np.integer))
            or not len(passes) <= num_train <= maximum):
        raise ValueError(f"GOTCHA num_train must be an integer in [{len(passes)}, {maximum}]")
    count, remainder = divmod(int(num_train), len(passes))
    pass_order = np.random.Generator(np.random.PCG64(42)).permutation(passes)
    extra = set(map(int, pass_order[:remainder]))
    order = np.random.Generator(np.random.PCG64(42)).permutation(np.arange(1, 361))
    return {p: tuple(sorted(map(int, order[:count + int(p in extra)]))) for p in passes}


def _npy_location(path: Path, member: str):
    """Locate one uncompressed C-order NPY without reading its array payload."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate NPZ members are forbidden')
        info = archive.getinfo(member)
        if info.compress_type != zipfile.ZIP_STORED or info.flag_bits & 1:
            raise ValueError('Native response must be uncompressed and unencrypted')
        # Read NPY header via a bounded raw-file stream. ZipExtFile may buffer
        # response bytes beyond the header, which a metadata-only plan must avoid.
        with path.open('rb') as stream:
            stream.seek(info.header_offset)
            header = stream.read(30)
            if len(header) != 30 or header[:4] != b'PK\x03\x04':
                raise ValueError('Invalid NPZ local header')
            filename_length, extra_length = struct.unpack_from('<HH', header, 26)
            start = info.header_offset + 30 + filename_length + extra_length
            stream.seek(start)
            version = np.lib.format.read_magic(stream)
            shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
            offset = stream.tell()
        if fortran or dtype.hasobject:
            raise ValueError('Native response must be C-order numeric data')
        size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if offset - start + size != info.file_size:
            raise ValueError('Response header and member size disagree')
        return tuple(shape), dtype, offset, int(info.CRC)


@dataclass(frozen=True)
class Observation:
    pass_id: int
    polarization: str
    sector_id: int
    pulse_index: int
    position_m: np.ndarray
    frequencies_hz: np.ndarray
    reference_range_m: float
    response: np.ndarray
    autofocus: str


class NativeShardReader:
    """Metadata-first native shard with role checks before every response seek."""
    def __init__(self, path, pass_id, polarization, *, roles=('train', 'validation'), train_sectors=None,
                 frequency_stride=1):
        from .gotcha_frequency_selection import NativeFrequencySelection, selection_config
        selection_config(frequency_stride)
        self.frequency_stride = frequency_stride
        self.path = Path(path)
        self.pass_id, self.polarization = int(pass_id), str(polarization).lower()
        selection((self.pass_id,), (self.polarization,))
        self.roles = frozenset(roles)
        if not self.roles or not self.roles <= {'train', 'validation'}:
            raise PermissionError('Training adapter only permits train/validation; test stays sealed')
        self._signature = self._stat()
        with np.load(self.path, allow_pickle=False) as source:
            self.metadata = json.loads(str(source['metadata_json'].item()))
            required = dict(schema=SOURCE_SCHEMA, pass_id=self.pass_id,
                            polarization=self.polarization,
                            shard_id=f'pass{self.pass_id}_{self.polarization}',
                            corrections_applied=False, native_frequency_preserved=True,
                            all_rows_role='train', source_file_count=360,
                            all_available_sectors_used=True, evaluation_holdout=False)
            for key, expected in required.items():
                if self.metadata.get(key) != expected:
                    raise ValueError(f'{self.path.name}: metadata.{key} must be {expected!r}')
            reference = self.metadata.get('phase_reference', {})
            for key, expected in dict(frequency_unit='Hz', position_unit='m', range_unit='m',
                                     reference_range_field='r0', frequency_values='native_stored_exact',
                                     geometry_contract='paired_monostatic_tx_equals_rx_same_observation').items():
                if reference.get(key) != expected:
                    raise ValueError(f'Native phase-reference {key} changed')
            names = ('frequencies_hz', 'x', 'y', 'z', 'r0', 'sector_id', 'pulse_index',
                     'pass_id', 'polarization', 'role', 'r_correct_raw', 'ph_correct_raw')
            self.arrays = {key: np.array(source[key], copy=True) for key in names}
        a = self.arrays
        n = a['sector_id'].size
        f = a['frequencies_hz']
        if f.ndim != 1 or f.size < 3 or not np.isfinite(f).all() or not np.all(np.diff(f.astype(np.float64)) > 0) or f.min() <= 0:
            raise ValueError('Native frequencies must be finite, positive and increasing')
        for key in ('x', 'y', 'z', 'r0', 'sector_id', 'pulse_index', 'pass_id', 'polarization', 'role'):
            if a[key].shape != (n,) or n == 0:
                raise ValueError(f'{key} must be a nonempty view-aligned vector')
        for key in ('x', 'y', 'z', 'r0'):
            if not np.isfinite(a[key]).all():
                raise ValueError(f'Nonfinite {key}')
        for key in ('sector_id', 'pulse_index', 'pass_id'):
            if a[key].dtype.kind not in 'iu':
                raise ValueError(f'{key} must be integral')
        if (set(a['sector_id'].tolist()) != set(range(1, 361))
                or not np.all(a['pass_id'] == self.pass_id)
                or not np.all(np.char.lower(a['polarization'].astype(str)) == self.polarization)
                or not np.all(a['role'] == 'train') or np.any(a['pulse_index'] < 0)):
            raise ValueError('Native shard identity/sector inventory does not match metadata')
        identities = np.stack((a['sector_id'], a['pulse_index']), axis=1)
        if np.unique(identities, axis=0).shape[0] != n:
            raise ValueError('Repeated sector/pulse identity')
        af = self.metadata.get('autofocus', {})
        self.co_pol = self.polarization in ('hh', 'vv')
        if af.get('applied') is not False or af.get('official_available') is not self.co_pol:
            raise ValueError('Autofocus provenance is absent or already applied')
        if self.co_pol:
            if af.get('source_shard_id') != f'pass{self.pass_id}_{self.polarization}' or af.get('mode') != 'source_af_unapplied':
                raise ValueError('Corrections must belong to the selected channel')
            for key in ('r_correct_raw', 'ph_correct_raw'):
                if a[key].shape != (n,) or not np.isfinite(a[key]).all():
                    raise ValueError('Invalid channel-owned autofocus arrays')
        elif af.get('mode') != 'official_af_absent' or a['r_correct_raw'].size or a['ph_correct_raw'].size:
            raise ValueError('Cross-polar channels must retain raw data without borrowed corrections')
        self.shape, self.dtype, self.offset, crc = _npy_location(self.path, 'response.npy')
        if self.shape != (n, len(f)) or self.dtype != np.dtype('complex64'):
            raise ValueError('Expected native complex64 [pulse, frequency] response')
        split = sector_split()
        if train_sectors is not None:
            selected_train = tuple(train_sectors)
            if (not selected_train or len(set(selected_train)) != len(selected_train)
                    or any(type(i) is not int or i not in split['train'] for i in selected_train)):
                raise ValueError('Training sectors must be distinct parent training sectors')
            selected_train = set(selected_train)
        else:
            selected_train = set(split['train'])
        self.row_roles = np.full(n, 'excluded', dtype='U10')
        self.sector_rows = {}
        for role, sectors in split.items():
            for sector in sectors:
                rows = np.flatnonzero(a['sector_id'] == sector)
                rows = rows[np.argsort(a['pulse_index'][rows])]
                rows.setflags(write=False)
                self.sector_rows[sector] = rows
                if role != 'train' or sector in selected_train:
                    self.row_roles[rows] = role
        self.row_roles.setflags(write=False)
        arrays_hash = hashlib.sha256()
        for key in sorted(a):
            arrays_hash.update(key.encode())
            arrays_hash.update(a[key].dtype.str.encode())
            arrays_hash.update(a[key].tobytes())
            a[key].setflags(write=False)
        self.identity = dict(pass_id=self.pass_id, polarization=self.polarization,
                             source_schema=SOURCE_SCHEMA, source_metadata_sha256=digest(self.metadata),
                             acquisition_sha256=arrays_hash.hexdigest(), response_crc32=crc,
                             response_shape=list(self.shape), response_dtype=self.dtype.str)
        self.frequencies_hz = f.astype(np.float64)
        self.frequencies_hz.setflags(write=False)
        self._frequency_selection = NativeFrequencySelection(self.frequencies_hz, frequency_stride)
        if self._stat() != self._signature:
            raise ValueError('Source archive changed during metadata preflight')
        self._mapping = None
        self.response_reads = 0

    def _stat(self):
        s = self.path.stat()
        return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns

    def frequencies_for_role(self, role):
        """Exact selected frequencies for every declared acquisition role."""
        return self._frequency_selection.frequencies(role)

    def frequency_indices_for_role(self, role):
        """Source-bin indices; source .shape/.frequencies_hz remain unchanged."""
        return self._frequency_selection.indices(role)

    def read(self, row: int) -> Observation:
        if isinstance(row, bool) or not isinstance(row, (int, np.integer)) or not 0 <= row < self.shape[0]:
            raise ValueError('Invalid native row')
        if self.row_roles[row] not in self.roles:
            raise PermissionError('Response row is outside the permitted role; test stays sealed')
        if self._stat() != self._signature:
            raise ValueError('Source archive changed after preflight')
        if self._mapping is None:
            self._mapping = np.memmap(self.path, mode='r', dtype=self.dtype, offset=self.offset, shape=self.shape)
        role = self.row_roles[row]
        indices = (self.frequency_indices_for_role(role)
                   if self.frequency_stride != 1 else slice(None))
        response = np.array(self._mapping[row, indices], dtype=np.complex128, copy=True)
        if not np.isfinite(response).all():
            raise ValueError('Nonfinite native response')
        self.response_reads += 1
        a = self.arrays
        r0 = float(np.float64(a['r0'][row]))
        autofocus = 'raw_official_af_absent'
        if self.co_pol:
            r0 += float(np.float64(a['r_correct_raw'][row]))
            response *= np.exp(1j * float(np.float64(a['ph_correct_raw'][row])))
            autofocus = 'own_published_source_af_once'
        return Observation(self.pass_id, self.polarization, int(a['sector_id'][row]),
                           int(a['pulse_index'][row]),
                           np.asarray([a[k][row] for k in ('x', 'y', 'z')], dtype=np.float64),
                           self.frequencies_for_role(role), r0, response, autofocus)


class GOTCHADataset:
    def __init__(self, root=DEFAULT_ROOT, *, shard_root=None, passes=range(1, 9),
                 polarizations=('hh',), region=None, roles=('train', 'validation'), num_train=None,
                 num_tx=1, num_rx=1, tx_indices=None, rx_indices=None, pulses_per_sector=0,
                 frequency_stride=1):
        from .gotcha_pulse_sampling import validate_pulse_limit, apply_training_selection
        from .gotcha_frequency_selection import selection_config, selection_contract, preflight
        selection_config(frequency_stride)
        self.frequency_stride = frequency_stride
        self.pulses_per_sector = validate_pulse_limit(pulses_per_sector)
        from .antenna_selection import selection as antenna_selection
        antenna_selection(num_tx, num_rx, tx_indices, rx_indices, source=(1, 1))
        self.num_tx = self.num_rx = 1
        self.tx_indices = self.rx_indices = (0,)
        self.root = Path(root)
        self.passes, self.polarizations = selection(passes, polarizations)
        selected_train = training_sectors(self.passes, num_train)
        self.num_train = sum(map(len, selected_train.values()))
        parent_split = sector_split()
        self.splits_by_pass = {p: {**parent_split, 'train': selected_train[p]} for p in self.passes}
        self.region = region or load_region()
        self.shard_root = Path(shard_root) if shard_root else self.root / 'New_Transfer' / 'shards'
        self.shards = {(p, pol): NativeShardReader(self.shard_root / f'pass{p}_{pol}.npz', p, pol,
                                                  roles=roles, train_sectors=selected_train[p],
                                                  frequency_stride=frequency_stride)
                       for p in self.passes for pol in self.polarizations}
        self.training_pulse_selection = apply_training_selection(self, self.pulses_per_sector)
        split_contract = {**catalog()['split'], 'sector_ids': parent_split}
        self.training_selection = None
        if self.num_train != 250 * len(self.passes):
            ids = [[p, s] for p in self.passes for s in selected_train[p]]
            self.training_selection = dict(schema='gotcha_nested_training_subset_v1',
                num_train=self.num_train, parent_num_train=250*len(self.passes), seed=42,
                selection='balanced_parent_permutation_prefix_seeded_pass_remainder',
                train_ids_sha256=digest(ids))
            split_contract = dict(schema='gotcha_nested_training_subset_v1', seed=42,
                unit='pass_sector', parent_schema=catalog()['split']['schema'],
                sector_ids_by_pass=self.splits_by_pass, training_selection=self.training_selection,
                same_holdout_roles_across_passes_and_polarizations=True,
                same_training_roles_across_polarizations=True, native_pulses_within_sector='all')
        # Legacy common-sector access stays available only for the parent split.
        self.split = parent_split if self.training_selection is None else self.splits_by_pass
        self.contract = dict(schema=SCHEMA, dataset_id=catalog()['dataset_id'],
                             passes=list(self.passes), polarizations=list(self.polarizations),
                             region=self.region.as_dict(), split=split_contract,
                             acquisition=[shard.identity for shard in self.shards.values()],
                             autofocus='published channel-owned HH/VV once; HV/VH raw',
                             phase_kernel='exp(-i*4*pi*f/c*(norm(x-antenna)-r0_effective))',
                             frequency_policy='native_stored_exact_no_resampling',
                             range_spreading='unit_native_phase_history',
                             transfer_training_checkpoints_reused=False)
        if self.training_pulse_selection is not None:
            split_contract['native_pulses_within_sector'] = 'same_fixed_cap_train_validation_test'
            self.contract['training_pulse_selection'] = self.training_pulse_selection
        self.frequency_selection = selection_contract(self.shards, frequency_stride)
        if self.frequency_selection is not None:
            self.contract['frequency_selection'] = self.frequency_selection
        # JSON-normalize tuples so a serialized resume identity compares exactly.
        self.contract = json.loads(json.dumps(self.contract))
        self.identity = digest(self.contract)
        self.frequency_preflight = preflight(self)

    def viewpoints(self, role):
        if role not in ('train', 'validation'):
            raise PermissionError('Test viewpoints are sealed from the training adapter')
        return [(p, sector) for p in self.passes for sector in self.splits_by_pass[p][role]]

    def observations(self, pass_id, sector_id, polarization):
        shard = self.shards[(pass_id, polarization)]
        if sector_id not in shard.sector_rows:
            raise ValueError('Unknown native sector')
        for row in shard.sector_rows[sector_id]:
            yield shard.read(int(row))

    def summary(self):
        result = dict(schema=SCHEMA, identity=self.identity, region=self.region.as_dict(),
                    passes=list(self.passes), polarizations=list(self.polarizations),
                    viewpoint_unit='pass_sector_with_all_native_pulses',
                    viewpoints={role:sum(len(s[role]) for s in self.splits_by_pass.values())
                                for role in ('train', 'validation', 'test')},
                    training_selection=self.training_selection,
                    pulses_by_polarization={pol:{role:sum(int(np.count_nonzero(self.shards[p,pol].row_roles == role))
                                                         for p in self.passes)
                                                for role in ('train', 'validation', 'test', 'excluded')}
                                            for pol in self.polarizations},
                    native_frequency_counts={f'pass{p}_{pol}':s.shape[1] for (p,pol),s in self.shards.items()},
                    response_payload_read=any(s.response_reads for s in self.shards.values()),
                    test_payload_accessible=False,
                    source_storage_roles='all-train labels superseded by this new, explicit split identity')
        if self.training_pulse_selection is not None:
            result.update(training_pulse_selection=self.training_pulse_selection,
                          viewpoint_unit='pass_sector_with_fixed_selected_pulses_in_each_role',
                          source_storage_roles='native archives unchanged; unselected pulses excluded in every role')
        if self.frequency_selection is not None:
            result.update(frequency_selection=self.frequency_selection,
                          selected_frequency_counts={f'pass{p}_{pol}': len(s.frequencies_for_role('train'))
                                                     for (p, pol), s in self.shards.items()},
                          frequency_preflight=self.frequency_preflight)
        return result


def validate_checkpoint(checkpoint: Mapping, dataset: GOTCHADataset, recipe: Mapping):
    if checkpoint.get('schema') != 'rift_gotcha_checkpoint_v1':
        raise ValueError('Not a new GOTCHA adapter checkpoint; historical checkpoints cannot resume here')
    if checkpoint.get('dataset_contract') != dataset.contract or checkpoint.get('dataset_identity') != dataset.identity:
        raise ValueError('Checkpoint dataset, region, passes, polarizations or split changed')
    if checkpoint.get('recipe') != dict(recipe):
        raise ValueError('Checkpoint training recipe changed')
