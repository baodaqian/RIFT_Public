"""RadarSplat-owned native GOTCHA conversion and released-model lifecycle.

No coherent renderer is substituted for RadarSplat. Native phase history is
converted to a sector matched-filter power image, then the same original
RadarSplat model/recipe used for RIFT is fitted. Source fields are read only by
the role-restricted public GOTCHA loader. See docs/RADARSPLAT_FIDELITY.md.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import signal

import numpy as np
import torch

from rift.gotcha_dataset import C, GOTCHADataset, Region, digest
from rift.radarsplat_b7873200 import RadarSplatGrid
from rift.radarsplat_b7873200_protocol import RECIPE_FILENAME, atomic_save_npz, atomic_write_json
from rift.radarsplat_release import COMMIT, SCHEMA, load_cuda_reference, reference_contract, model_recipe, PROFILES
from rift import radarsplat_release_training as engine

CACHE_SCHEMA = "rift_radarsplat_gotcha_mf_power_v1"
CONTROL_SCHEMA = "rift_radarsplat_gotcha_run_v1"
CONTROL_FILE = "radarsplat_gotcha.pt"
DEFAULTS = dict(azimuth_samples=33, elevation_samples=33, point_chunk=128, frequency_chunk=256)


def adapter_config(config=None):
    config = {} if config is None else config
    if not isinstance(config, dict) or set(config)-set(DEFAULTS)-{"fidelity_profile"}:
        raise ValueError("RadarSplat GOTCHA config accepts acquisition sampling, execution chunks and fidelity_profile; the model recipe is fixed within each profile")
    if config.get("fidelity_profile", "budget48") not in PROFILES:
        raise ValueError("Unknown RadarSplat scene-budget profile")
    value = {**DEFAULTS, **{k:v for k,v in config.items() if k != "fidelity_profile"}}
    if any(type(v) is not int or v < 1 for v in value.values()):
        raise ValueError("RadarSplat GOTCHA adapter counts must be positive integers")
    if value["azimuth_samples"] < 11 or value["azimuth_samples"] % 2 == 0 or value["elevation_samples"] < 3 or value["elevation_samples"] % 2 == 0:
        raise ValueError("Use odd azimuth >=11 and elevation >=3 sample counts")
    return value


def frame_for_positions(positions):
    """Mean native sensor position; target at azimuth pi to avoid the seam."""
    center = np.asarray(positions, dtype=np.float64).mean(0)
    distance = np.linalg.norm(center)
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError("Sector sensor must be outside the region centre")
    x = center/distance
    up = np.array([0., 0., 1.]) if abs(x[2]) < .99 else np.array([0., 1., 0.])
    y = np.cross(up, x); y /= np.linalg.norm(y)
    z = np.cross(x, y)
    pose = np.eye(4)
    pose[:3, :3] = np.stack((x, y, z), 1)
    pose[:3, 3] = center
    return pose


def sampling(dataset, config):
    e = float(dataset.region.half_extent_m)
    rayleigh = min(C/(2*(s.frequencies_hz[-1]-s.frequencies_hz[0])) for s in dataset.shards.values())
    radius = math.sqrt(3)*e
    # Include the origin and cover the entire cube's circumscribed sphere;
    # sample range at least twice per finest native Rayleigh interval.
    count = max(33, int(math.ceil(4*radius/rayleigh))+1)
    count += (count % 2 == 0)
    return dict(scene_center_m=[0., 0., 0.], scene_extent_m=e, range_half_span_m=radius,
                n_range=count, n_azimuth=config["azimuth_samples"], n_elevation=config["elevation_samples"],
                range_resolution_m=2*radius/(count-1), finest_native_rayleigh_m=rayleigh,
                intermediate_azimuth_factor=10,
                range_filter_policy="original Gaussian kernel with sigma equal to native Rayleigh range; integer source discretization",
                azimuth_filter_policy="wavelength/(2*sector aperture), truncated on original sigma-four-pixel kernel; one-pulse fallback is ROI angular span")


class GOTCHAPowerCache:
    """One polarization's source-bound targets; constructor reads metadata only."""
    local_azimuth = True
    def __init__(self, dataset, polarization, root, config=None):
        if polarization not in dataset.polarizations:
            raise ValueError("Polarization is outside the selected native acquisition")
        self.dataset, self.polarization, self.root = dataset, polarization, Path(root)
        self.config = adapter_config(config)
        self.grid = sampling(dataset, self.config)
        self.identity = dict(schema="rift_radarsplat_gotcha_head_v1", dataset_contract=dataset.contract,
                             dataset_identity=dataset.identity, polarization=polarization)
        self.recipe = dict(schema=CACHE_SCHEMA, sealed_protocol_identity=self.identity, config=self.config,
            grid=self.grid, phase="sum_pulse mean_native_frequency(response * exp(+i*4*pi*f*(distance-r0_effective)/c)); divide by pulse count",
            power="sum_elevation(abs(coherent_sector_MF)**2)",
            response_selection=("same fixed native pulse cap in train/validation/test sectors"
                                if dataset.training_pulse_selection else
                                "all native pulses in registered train/validation sectors"),
            normalization="one training-only peak per polarization across all selected passes",
            source_autofocus="public ingress applies channel-owned correction once; cross-pol raw",
            pose="metadata mean sector position; outward x-axis places region at azimuth pi; exact native pulse positions remain in MF",
            raster_domain="local full-circle lattice crop with complete original filter halo",
            filter_approximation="Gaussian source sensor kernel approximates SAR power PSF; not sensor equivalence")
        from rift.gotcha_frequency_selection import recipe_fields
        acquisition_fields = recipe_fields(dataset)
        if dataset.frequency_selection:
            self.recipe.update(acquisition_fields,
                response_selection="same fixed pulse cap and native-bin rule for train/validation/test; test payload sealed",
                phase="sum_pulse mean_role_frequency(response * exp(+i*4*pi*f*(distance-r0_effective)/c)); divide by role pulse count")
        self.recipe_digest = digest(self.recipe)
        self.view_keys, self.roles, self.calibration = {}, {}, {}
        positions = []
        self.train_indices, self.validation_indices = [], []
        for role, destination in (("train", self.train_indices), ("validation", self.validation_indices)):
            for pass_id, sector in dataset.viewpoints(role):
                index = len(self.view_keys)
                self.view_keys[index], self.roles[index] = (pass_id, sector), role
                destination.append(index)
                shard = dataset.shards[pass_id, polarization]
                rows = shard.sector_rows[sector]
                if not len(rows):
                    raise ValueError("Empty native pass-sector")
                a = shard.arrays
                xyz = dataset.region.to_local(np.stack([a[k][rows] for k in ("x", "y", "z")], -1))
                pose = frame_for_positions(xyz)
                distance = np.linalg.norm(pose[:3, 3])
                radius = self.grid["range_half_span_m"]
                if distance <= radius:
                    raise ValueError("Native sensor lies inside the selected scene support")
                half_angle = math.asin(radius/distance)
                ranges = np.linspace(distance-radius, distance+radius, self.grid["n_range"])
                az = np.linspace(math.pi-half_angle, math.pi+half_angle, self.grid["n_azimuth"])
                el = np.linspace(-half_angle, half_angle, self.grid["n_elevation"])
                daz = math.degrees(float(az[1]-az[0]))
                aperture = float(np.linalg.norm(xyz-xyz[0], axis=1).max())
                beam = math.degrees(C/float(shard.frequencies_hz.mean())/(2*aperture)) if aperture > 0 else 2*math.degrees(half_angle)
                beam = min(360., max(beam, 2*daz/10))
                rayleigh = C/(2*(shard.frequencies_hz[-1]-shard.frequencies_hz[0]))
                grid = RadarSplatGrid(num_range_bins=len(ranges), range_resolution_m=self.grid["range_resolution_m"],
                    range_start_m=float(ranges[0])-self.grid["range_resolution_m"]/2,
                    azimuth_start_deg=math.degrees(float(az[0]))-daz/2, azimuth_span_deg=len(az)*daz,
                    intermediate_azimuth_resolution_deg=daz/10, output_azimuth_resolution_deg=daz,
                    azimuth_beamwidth_deg=beam, spectral_leakage_width_m=6*rayleigh)
                self.calibration[index] = dict(sensor_to_world=pose, range_m=ranges, azimuth_rad=az, elevation_rad=el,
                                               renderer_grid=asdict(grid), native_pulse_count=len(rows))
                positions.append(pose[:3, 3].tolist())
        self.train_indices, self.validation_indices = tuple(self.train_indices), tuple(self.validation_indices)
        self.acquisition_record = dict(schema=CACHE_SCHEMA, dataset_contract=dataset.contract, polarization=polarization,
            view_indices=list(self.view_keys), viewpoint_positions=positions,
            view_keys=[list(k) for k in self.view_keys.values()], grid=self.grid,
            calibration=[{k:(v.tolist() if isinstance(v, np.ndarray) else v) for k,v in row.items()} for row in self.calibration.values()])
        self.is_development_subset = False
        self.manifest = dict(schema=CACHE_SCHEMA, recipe_digest=self.recipe_digest, targets={})
        if (self.root/RECIPE_FILENAME).exists():
            if json.loads((self.root/RECIPE_FILENAME).read_text()) != self.recipe:
                raise ValueError("GOTCHA target cache source/region/roles/polarization/recipe mismatch")
        elif self.root.exists() and any(self.root.iterdir()):
            raise ValueError("Unidentified GOTCHA target cache")
        if (self.root/"manifest.json").exists():
            self.manifest = json.loads((self.root/"manifest.json").read_text())
            if (self.manifest.get("schema") != CACHE_SCHEMA or self.manifest.get("recipe_digest") != self.recipe_digest
                    or not isinstance(self.manifest.get("targets"), dict)
                    or set(self.manifest["targets"])-set(map(str, self.view_keys))):
                raise ValueError("GOTCHA target manifest identity mismatch")
        self.train_peak_power = None
        if (self.root/"stats.json").exists():
            stats = json.loads((self.root/"stats.json").read_text())
            expected = {str(i): self.manifest["targets"].get(str(i)) for i in self.train_indices}
            peak = stats.get("train_peak_power")
            if (stats.get("recipe_digest") != self.recipe_digest or stats.get("training_targets") != expected
                    or any(v is None for v in expected.values()) or not isinstance(peak, (int,float))
                    or not math.isfinite(peak) or peak <= 0):
                raise ValueError("GOTCHA normalization statistics are not bound to the training targets")
            self.train_peak_power = float(peak)

    def target_path(self, index):
        return self.root/"targets"/f"view_{index:05d}.npz"

    def renderer_grid(self, index):
        return RadarSplatGrid(**self.calibration[index]["renderer_grid"])

    def read_target(self, index, role, *, allow_unrecorded=False):
        if role not in ("train", "validation") or self.roles.get(index) != role:
            raise PermissionError("GOTCHA target is outside the requested sealed role")
        path = self.target_path(index)
        expected = self.manifest["targets"].get(str(index))
        if expected is None and not allow_unrecorded:
            raise ValueError("Uncommitted GOTCHA target")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected is not None and actual != expected:
            raise ValueError("GOTCHA target bytes changed")
        with np.load(path, allow_pickle=False) as source:
            arrays = {k:source[k] for k in source.files}
        if (str(arrays.get("recipe_digest")) != self.recipe_digest or str(arrays.get("role")) != role
                or int(arrays.get("view_index", -1)) != index):
            raise ValueError("GOTCHA target identity mismatch")
        for key in ("sensor_to_world", "range_m", "azimuth_rad", "elevation_rad"):
            expected_axis = self.calibration[index][key]
            if key not in arrays or arrays[key].dtype != expected_axis.dtype or not np.array_equal(arrays[key], expected_axis):
                raise ValueError(f"GOTCHA target calibration changed: {key}")
        power = arrays.get("radarsplat_mf_power")
        if (power is None or power.dtype != np.float32 or power.shape != (self.grid["n_azimuth"], self.grid["n_range"])
                or not np.isfinite(power).all() or np.any(power < 0)):
            raise ValueError("Invalid GOTCHA MF power target")
        return arrays

    def prepare(self, *, device, should_stop=lambda:False):
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root/"targets").mkdir(exist_ok=True)
        atomic_write_json(self.root/RECIPE_FILENAME, self.recipe)
        peak = 0.
        for role, indices in (("train", self.train_indices), ("validation", self.validation_indices)):
            for index in indices:
                if should_stop():
                    return False
                path = self.target_path(index)
                if not path.exists():
                    arrays = self.calibration[index]
                    power = sector_power(self.dataset, *self.view_keys[index], self.polarization,
                                         arrays, self.config, device=device)
                    atomic_save_npz(path, recipe_digest=np.asarray(self.recipe_digest), role=np.asarray(role),
                        view_index=np.asarray(index), radarsplat_mf_power=power,
                        **{k:arrays[k] for k in ("sensor_to_world", "range_m", "azimuth_rad", "elevation_rad")})
                arrays = self.read_target(index, role, allow_unrecorded=True)
                self.manifest["targets"][str(index)] = hashlib.sha256(path.read_bytes()).hexdigest()
                atomic_write_json(self.root/"manifest.json", self.manifest)
                if role == "train":
                    peak = max(peak, float(arrays["radarsplat_mf_power"].max()))
            if role == "train":
                if peak <= 0 or not math.isfinite(peak):
                    raise ValueError("Training-only GOTCHA power peak must be positive")
                if self.train_peak_power is not None and self.train_peak_power != peak:
                    raise ValueError("GOTCHA training normalization changed")
                self.train_peak_power = peak
                atomic_write_json(self.root/"stats.json", dict(recipe_digest=self.recipe_digest, train_peak_power=peak,
                    training_targets={str(i):self.manifest["targets"][str(i)] for i in self.train_indices}))
        return True


@torch.no_grad()
def sector_power(dataset, pass_id, sector, polarization, calibration, config, *, device):
    """Exact ragged-frequency adjoint before taking power; no FFT resampling."""
    from rift.radarsplat_fidelity import polar_world_points
    points = torch.as_tensor(polar_world_points(calibration), dtype=torch.float64, device=device)
    coherent = torch.zeros(len(points), dtype=torch.complex128, device=device)
    count = 0
    for observation in dataset.observations(pass_id, sector, polarization):
        antenna = torch.as_tensor(dataset.region.to_local(observation.position_m), dtype=torch.float64, device=device)
        f = torch.as_tensor(observation.frequencies_hz.copy(), dtype=torch.float64, device=device)
        y = torch.as_tensor(observation.response, dtype=torch.complex128, device=device)
        for begin in range(0, len(points), config["point_chunk"]):
            distance = (points[begin:begin+config["point_chunk"]]-antenna).norm(dim=-1)-observation.reference_range_m
            value = torch.zeros(len(distance), dtype=torch.complex128, device=device)
            for fi in range(0, len(f), config["frequency_chunk"]):
                phase = (4*math.pi/C)*distance[:, None]*f[None, fi:fi+config["frequency_chunk"]]
                value += (torch.exp(1j*phase)*y[None, fi:fi+config["frequency_chunk"]]).sum(-1)
            coherent[begin:begin+len(value)] += value/len(f)
        count += 1
    if count != calibration["native_pulse_count"]:
        raise ValueError("Native sector pulse count changed during conversion")
    shape = (len(calibration["elevation_rad"]), len(calibration["azimuth_rad"]), len(calibration["range_m"]))
    power = (coherent/count).abs().square().reshape(shape).sum(0)
    if not torch.isfinite(power).all():
        raise ValueError("Nonfinite native GOTCHA MF target")
    return power.cpu().numpy().astype(np.float32)


def planning(dataset, config=None):
    profile = (config or {}).get("fidelity_profile", "budget48")
    config = adapter_config(config)
    grid = sampling(dataset, config)
    return dict(schema=CONTROL_SCHEMA, source_commit=COMMIT, model_schema=SCHEMA,
        model_recipe=model_recipe(profile), dataset_contract=dataset.contract,
        dataset_identity=dataset.identity, adapter_config=config, grid=grid,
        heads=list(dataset.polarizations), model_updates_per_head=2000,
        head_policy="independent polarization scenes, each shared jointly across selected passes",
        fidelity_status="released_source_with_native_MF_conversion_cuda_unvalidated")


def _validate_head_checkpoint(path, cache, profile="budget48"):
    """Metadata and raw state gate, before conversion or target reads."""
    import train_radarsplat as lifecycle
    saved = lifecycle._load_checkpoint(path, torch.device("cpu"))
    from rift.radarsplat_release import intensity_mapping_from_identity
    mapping = intensity_mapping_from_identity(saved.get("identity") or {})
    if (cache.train_peak_power is None or saved.get("schema") != SCHEMA
            or saved.get("identity") != engine.identity_for_cache(cache, profile, mapping)):
        raise ValueError("GOTCHA RadarSplat head checkpoint identity mismatch")
    if not lifecycle._directly_equal(saved.get("acquisition_record"), cache.acquisition_record):
        raise ValueError("GOTCHA RadarSplat head acquisition mismatch")
    splats, optimizers = engine.create_scene(scene_scale=saved["identity"]["adapter"]["initialization_scene_scale"],
                                            scene_center=np.zeros(3), device="cpu", num_points=saved["identity"]["model_recipe"]["init_num_pts"])
    engine.restore_state(saved, splats, optimizers, engine.position_scheduler(optimizers),
                         lifecycle.DeterministicViewSampler(cache.train_indices, 42))
    return saved


def run_gotcha(*, dataset, output_dir, config, device, resume):
    """Root backend; all mutations occur only in an execution allocation."""
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("RadarSplat GOTCHA conversion/fitting requires an experiment-manager allocation")
    import train_radarsplat as lifecycle
    output = Path(output_dir)
    identity = planning(dataset, config)
    control = output/CONTROL_FILE
    state = dict(schema=CONTROL_SCHEMA, identity=identity, completed_heads=[])
    if resume is not None:
        if Path(resume).resolve() != control.resolve():
            raise ValueError(f"Resume must select this run's {CONTROL_FILE}")
        state = lifecycle._load_checkpoint(control, torch.device("cpu"))
        if state.get("schema") != CONTROL_SCHEMA or state.get("identity") != identity:
            raise ValueError("GOTCHA RadarSplat run identity mismatch before response access")
        completed = state.get("completed_heads")
        if not isinstance(completed, list) or completed != list(dataset.polarizations)[:len(completed)]:
            raise ValueError("Invalid completed polarization prefix")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Existing GOTCHA RadarSplat output requires its run checkpoint or a new output root")
    caches = {pol:GOTCHAPowerCache(dataset, pol, output/pol/"targets", config) for pol in dataset.polarizations}
    results = {}
    # Check every existing head before preparing any channel. A changed channel
    # cannot be discovered only after another channel has read native responses.
    for pol, cache in caches.items():
        folder = output/pol/"checkpoints"
        latest = folder/"checkpoint_latest.pt"
        if latest.exists():
            saved = _validate_head_checkpoint(latest, cache, config.get("fidelity_profile", "budget48"))
            if pol in state["completed_heads"] and saved["step"] != 2000:
                raise ValueError("Completed GOTCHA head lacks a final source step")
            if pol in state["completed_heads"]:
                final = folder/"checkpoint_final.pt"
                if not final.exists() or not lifecycle._directly_equal(lifecycle._load_checkpoint(final, torch.device("cpu")), saved):
                    raise ValueError("Completed GOTCHA head final/recovery checkpoints disagree")
                summary_path = folder/"summary.json"
                expected = dict(identity=saved["identity"], step=2000, validation=saved.get("validation"))
                if not isinstance(saved.get("validation"), dict) or not summary_path.exists() or json.loads(summary_path.read_text()) != expected:
                    raise ValueError("Completed GOTCHA head summary is missing or inconsistent")
                results[pol] = dict(step=2000, validation=saved["validation"], checkpoint=str(final))
        elif pol in state["completed_heads"] or (folder.exists() and any(folder.iterdir())):
            raise ValueError("GOTCHA head state lacks its recovery checkpoint")
    rendering, ssim = load_cuda_reference(device=device)
    output.mkdir(parents=True, exist_ok=True)
    lifecycle._atomic_torch_save(control, state)
    atomic_write_json(output/"source.json", dict(root=str(dataset.root.resolve()), shard_root=str(dataset.shard_root.resolve()),
        passes=list(dataset.passes), polarizations=list(dataset.polarizations), region=dataset.region.as_dict(),
        num_train=len(dataset.viewpoints("train"))))
    stop = [False]
    previous = {s:signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    for s in previous:
        signal.signal(s, lambda *_:stop.__setitem__(0, True))
    try:
        for pol, cache in caches.items():
            if pol in state["completed_heads"]:
                continue
            if not cache.prepare(device=device, should_stop=lambda:stop[0]):
                return dict(status="interrupted", resume=str(control))
            results[pol] = engine.train(cache, output/pol/"checkpoints", device=torch.device(device),
                                       resume=True, rendering=rendering, fused_ssim=ssim, profile=config.get("fidelity_profile", "budget48"))
            state["completed_heads"].append(pol)
            lifecycle._atomic_torch_save(control, state)
        return dict(status="complete", resume=str(control), results=results,
                    heads=list(dataset.polarizations), observable="clipped normalized native MF power")
    except SystemExit as exc:
        if exc.code == 143:
            return dict(status="interrupted", resume=str(control))
        raise
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)


def cache_from_run(run_root, polarization, *, dataset_root=None, shard_root=None):
    """Rebind a stored readout to current native metadata without payload access."""
    root = Path(run_root)
    source = json.loads((root/"source.json").read_text())
    region = Region(**source["region"])
    selected_shards = shard_root or (Path(dataset_root)/"New_Transfer/shards" if dataset_root else source["shard_root"])
    import train_radarsplat as lifecycle
    from rift.gotcha_pulse_sampling import pulse_limit_from_contract
    from rift.gotcha_frequency_selection import kwargs_from_contract
    state = lifecycle._load_checkpoint(root/CONTROL_FILE, torch.device("cpu"))
    pulses_per_sector = pulse_limit_from_contract(state["identity"]["dataset_contract"])
    dataset = GOTCHADataset(dataset_root or source["root"], shard_root=selected_shards,
                            passes=source["passes"], polarizations=source["polarizations"], region=region,
                            num_train=source.get("num_train"), pulses_per_sector=pulses_per_sector,
                            **kwargs_from_contract(state["identity"]["dataset_contract"]))
    config = dict(state["identity"]["adapter_config"])
    from rift.radarsplat_release import profile_from_identity
    config["fidelity_profile"] = profile_from_identity(state["identity"])
    if state.get("schema") != CONTROL_SCHEMA or state.get("identity") != planning(dataset, config):
        raise ValueError("GOTCHA readout source/region/recipe changed")
    return GOTCHAPowerCache(dataset, polarization, root/polarization/"targets", config)
