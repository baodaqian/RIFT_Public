"""Pinned original RadarSplat implementation, with explicit sensor adapters.

The model is loaded from the authors' source, not the independent Torch port.
The default recipe follows run_all_radarsplat.sh -> run_radarsplat.sh (default).
See protocols/radarsplat_official_reference.json and docs/RADARSPLAT_FIDELITY.md.
"""
from __future__ import annotations

import ast
import importlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "protocols/radarsplat_official_reference.json"
COMMIT = "ea9c8f530c708622cc3b1b560436b5557ac6a49b"
REFERENCE_ROOT = ROOT / "external/radarsplat_reference" / COMMIT
SCHEMA = "rift_radarsplat_released_model_v1"
PROFILES = {"upstream": 20_000, "budget48": 112_000}
# Image intensity domain. The release trains on Navtech polar images, whose bytes are
# received power converted to dB and quantised to 0..255, divided by 255 (a log-power
# domain in which its absolute thresholds -- occupancy .10, raster 1/255 -- are set).
# LINEAR (power / TRAIN peak) is the earlier conversion and what a recipe without the
# key means. LOG maps TRAIN-peak-relative power over 60 dB to [0, 1], the Radar Fields
# adapter's mapping (radar_fields_dataset.normalize_power_db); the vendor's dB per count
# is not published, so the 60 dB span is our declared choice (user decision 3,
# 2026-09-22). The target cache stores raw MF power and is identical under both.
LINEAR_INTENSITY = "linear_train_peak_v1"
LOG_INTENSITY = "log_train_peak_60db_v1"
INTENSITY_MAPPINGS = (LINEAR_INTENSITY, LOG_INTENSITY)
DEFAULT_INTENSITY = LOG_INTENSITY
LOG_DYNAMIC_RANGE_DB = 60.0


def intensity(power, train_peak, mapping=LINEAR_INTENSITY):
    """Native MF power -> the image intensity the recipe trains on (NumPy or Torch).

    LINEAR returns power / peak unclipped, exactly the earlier expression; its
    callers clip where they always did. LOG is bounded to [0, 1] by construction.
    """
    if mapping == LINEAR_INTENSITY:
        return power/train_peak
    if mapping != LOG_INTENSITY:
        raise ValueError(f"unknown RadarSplat intensity mapping {mapping!r}")
    if not math.isfinite(float(train_peak)) or float(train_peak) <= 0:
        raise ValueError("TRAIN peak power must be positive")
    floor = 10.0**(-LOG_DYNAMIC_RANGE_DB/10)
    if torch.is_tensor(power):
        db = 10*torch.log10((power/train_peak).clamp_min(floor))
        return ((db+LOG_DYNAMIC_RANGE_DB)/LOG_DYNAMIC_RANGE_DB).clamp(0, 1)
    db = 10*np.log10(np.maximum(np.asarray(power, dtype=np.float64)/train_peak, floor))
    return np.clip((db+LOG_DYNAMIC_RANGE_DB)/LOG_DYNAMIC_RANGE_DB, 0, 1)


def intensity_mapping_from_identity(identity):
    """A saved identity's mapping; identities from before the key are LINEAR."""
    mapping = identity.get("intensity_mapping", LINEAR_INTENSITY)
    if mapping not in INTENSITY_MAPPINGS:
        raise ValueError(f"unknown RadarSplat intensity mapping {mapping!r}")
    return mapping


def model_recipe(profile="upstream"):
    """Keep the pinned launcher immutable; disclose the user's count override."""
    if profile not in PROFILES:
        raise ValueError("Unknown released RadarSplat scene-budget profile")
    value = dict(reference_contract()["launch_recipe"])
    value["init_num_pts"] = PROFILES[profile]
    return value


def profile_from_identity(identity):
    count = identity.get("model_recipe", {}).get("init_num_pts")
    for profile, expected in PROFILES.items():
        if type(count) is int and count == expected:
            return profile
    raise ValueError("Unknown RadarSplat checkpoint Gaussian-count recipe")


def reference_contract():
    return json.loads(MANIFEST.read_text())


def verify_reference(root=REFERENCE_ROOT, *, cuda_dependencies=False):
    """Check required files without hashing trusted GitHub source bytes."""
    root = Path(root).resolve()
    contract = reference_contract()
    # Historical digest values are provenance only; use their file inventory.
    for relative in contract["files_sha256"]:
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"Missing RadarSplat source: {path}; "
                               "run scripts/fetch_radarsplat_reference.py")
    if cuda_dependencies:
        for relative in contract["glm_files_sha256"]:
            path = root / "gsplat/cuda/csrc/third_party/glm" / relative
            if not path.is_file():
                raise RuntimeError(f"Missing GLM source: {path}")
    return root


def source_functions(relative, names, namespace, *, root=REFERENCE_ROOT):
    """Load unmodified pure function ASTs without upstream GUI/data side effects.

    Only used for released initialization and preprocessing. The production
    rasterizer is imported normally, including its original CUDA implementation.
    """
    root = verify_reference(root)
    path = root / relative
    tree = ast.parse(path.read_text())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in selected} != set(names):
        raise ValueError("Pinned source function inventory changed")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def create_scene(*, scene_scale, scene_center, device="cuda", seed=42,
                 root=REFERENCE_ROOT, num_points=20_000):
    """Execute the original initializer (including its two scale factors).

    num_points is bound by the selected profile (20,000 upstream or 112,000
    budget48), with smaller counts for synthetic tests. Units are adapter units.
    """
    ns = dict(torch=torch, np=np, math=math, cfg=SimpleNamespace(init_scale=.5))
    source_functions("examples/utils.py", ["rgb_to_sh"], ns, root=root)
    source_functions("examples/radar_simple_trainer.py", ["create_splats_with_optimizers"], ns, root=root)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return ns["create_splats_with_optimizers"](
            parser=None, init_type="random", init_num_pts=num_points, init_extent=1.,
            init_opacity=.5, init_scale=.5, scene_scale=float(scene_scale),
            scene_center=np.asarray(scene_center, dtype=np.float64), sh_degree=5,
            sparse_grad=False, visible_adam=False, batch_size=1, device=str(device))


def position_scheduler(optimizers):
    return torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=.01**(1/2000))


def load_cuda_reference(root=REFERENCE_ROOT, *, device="cuda"):
    """Require the actual upstream CUDA renderer and pinned fused SSIM.

    Never substitute the local Torch rasterizer or an installed vanilla gsplat.
    Source is selected before import; an already imported different copy fails.
    """
    root = verify_reference(root)
    if not torch.cuda.is_available():
        raise RuntimeError("Released RadarSplat requires CUDA; no CPU/alternate-model fallback is allowed")
    selected_device = torch.device(device)
    if selected_device.type != "cuda":
        raise ValueError("The original RadarSplat implementation requires a CUDA device")
    torch.cuda.set_device(selected_device.index if selected_device.index is not None else torch.cuda.current_device())
    for name in ("gsplat",):
        previous = sys.modules.get(name)
        if previous is not None and not Path(previous.__file__).resolve().is_relative_to(root):
            raise RuntimeError("Another gsplat was already imported; use a fresh process for RadarSplat")
    verify_reference(root, cuda_dependencies=True)
    sys.path.insert(0, str(root))
    rendering = importlib.import_module("gsplat.rendering")
    return rendering, importlib.import_module("fused_ssim").fused_ssim


def recipe(*, dataset_identity, target_recipe, train_peak, half_extent_m, profile="upstream",
           intensity_mapping=LINEAR_INTENSITY):
    """A fixed model recipe and a separately enumerated dataset conversion.

    LINEAR leaves the identity byte-identical to recipes from before the key.
    """
    if not math.isfinite(half_extent_m) or half_extent_m <= 0:
        raise ValueError("Scene extent must be positive")
    if intensity_mapping not in INTENSITY_MAPPINGS:
        raise ValueError(f"unknown RadarSplat intensity mapping {intensity_mapping!r}")
    contract = reference_contract()
    identity = dict(schema=SCHEMA, source_commit=COMMIT, dataset_identity=dataset_identity,
                target_recipe=target_recipe, train_peak_power=float(train_peak),
                model_recipe=model_recipe(profile), seed=42,
                adapter=dict(schema="rift_radarsplat_sensor_adapter_v2",
                    half_extent_m=float(half_extent_m), model_units_per_m=50/half_extent_m,
                    initialization_scene_scale=110.,
                    source_scene_scale_margin=1.1,
                    coordinate_units="registered scene cube maps to the release's 100-unit sensing diameter",
                    observable="clipped train-peak-normalized native MF power",
                    occupancy="11 nearest training directions; released denoiser; mean then threshold 0.10",
                    multipath="released detector/periodic fit on training images; nearest training direction reprojection",
                    fft_amplitude_conversion="unnormalized FFT threshold 30 scaled by N / floor(50 / 0.0596); no fitted threshold",
                    image_sampling="full-circle upstream raster/filter, calibrated local range crop and output-bin sampling",
                    sensor_filters="dataset range resolution, beamwidth and leakage width; original filter formula",
                    optimizer_sampler="seeded shuffled training-view cycles; fixed 2000 updates",
                    validation="sealed validation only, final source step; reserved test excluded"))
    if intensity_mapping != LINEAR_INTENSITY:
        identity["intensity_mapping"] = intensity_mapping
        identity["adapter"]["observable"] = ("train-peak-relative native MF power mapped over 60 dB to [0, 1] "
                                             "(the release's dB-quantised image domain)")
    return identity


def release_loss(power, occupancy, target, labels, splats, fused_ssim):
    """Literal released default-launcher objective, including scale clamp."""
    l1 = F.l1_loss(power, target)
    occ = F.l1_loss(occupancy, labels.to(occupancy))
    ssim = 1-fused_ssim(power[None, None].repeat(1, 3, 1, 1),
                       target[None, None].repeat(1, 3, 1, 1), padding="valid")
    size = torch.relu(torch.exp(torch.clamp(splats["scales"], max=10.))-1.).mean()
    probability = torch.relu(splats["opacities"].sigmoid()+splats["noise_probs"].sigmoid()-1).mean()
    total = .8*l1 + .2*ssim + 10*occ + 100*size + 1000*probability
    return dict(total=total, power_l1=l1, occupancy_l1=occ, ssim=ssim,
                max_size=size, opacity_noise=probability)


class ReleasedRenderer:
    """Calls the original renderer and filters; adapts only image coordinates.

    The upstream covariance/SH conventions, per-product cutoff, probability and
    power clipping and effective SH degree cap are retained verbatim.
    """
    def __init__(self, rendering, units_per_m, *, local_azimuth=False):
        self.source, self.units = rendering, float(units_per_m)
        self.local_azimuth = local_azimuth

    def __call__(self, splats, pose, grid, active_degree, multipath):
        device = splats["means"].device
        pose = pose.to(device=device, dtype=torch.float32).clone()
        pose[:3, 3] *= self.units
        # The original filter derives spacing from a complete 360-degree image.
        # Round up the sensor sampling lattice, then sample the requested bins.
        stride = grid.azimuth_stride
        height = math.ceil(360/(grid.intermediate_azimuth_resolution_deg*stride))*stride
        old_deg = 360/height
        new_deg = math.nextafter(stride*old_deg, math.inf)
        az = grid.azimuth_start_deg + (torch.arange(grid.output_azimuth_bins, device=device)+.5)*grid.output_azimuth_resolution_deg
        row = (torch.remainder(az-.5*old_deg, 360)/new_deg)
        azimuth_offset, filter_resolution, filter_beam = 0, new_deg, grid.azimuth_beamwidth_deg
        if self.local_azimuth:
            # The native GOTCHA ROI subtends a tiny angle. Evaluate a contiguous
            # crop of the same full-circle pixel lattice, with enough halo that
            # the original kernel's circular boundary never reaches the ROI.
            # Rescale only the helper's degree arguments: its old_resolution is
            # hard-coded as 360/H. This retains exactly its pixel kernel/stride.
            if float(row.max()-row.min()) > grid.output_azimuth_bins+2:
                raise ValueError("Local RadarSplat target crosses the circular seam")
            window = int(grid.azimuth_beamwidth_deg/old_deg)
            window += (window % 2 == 0)
            padding = math.ceil((window//2)/stride)+1
            first = math.floor(float(row.min()))-padding
            last = math.ceil(float(row.max()))+padding+1
            height = (last-first)*stride
            azimuth_offset = first*stride
            row = row-first
            virtual_old = 360/height
            filter_resolution = math.nextafter(stride*virtual_old, math.inf)
            filter_beam = (window+.25)*virtual_old
        dr = grid.range_resolution_m*self.units
        leakage = grid.spectral_leakage_width_m*self.units
        halo = int(((leakage/2)/dr)/3)*3
        width = grid.num_range_bins+2*halo
        start = grid.range_start_m*self.units-halo*dr
        k = pose.new_tensor([[1/dr, 0, -start/dr], [0, 1/math.radians(old_deg), -azimuth_offset], [0, 0, 1]])
        products = self.source._radar_rasterization(
            means=splats["means"], quats=splats["quats"], scales=splats["scales"].exp(),
            opacities=splats["opacities"].sigmoid(), noise_probs=splats["noise_probs"].sigmoid(),
            colors=torch.cat((splats["sh0"], splats["shN"]), 1),
            viewmats=pose[None], Ks=k[None], width=width, height=height,
            near_plane=-10., far_plane=10., sh_degree=active_degree, packed=False,
            rasterize_mode="classic", distributed=False, camera_model="ortho", sph=True)
        # Native raw pixel centres are at half integers. Filtering samples row
        # zero, then every stride. Periodic interpolation preserves this offset.
        r = torch.arange(grid.num_range_bins, device=device, dtype=row.dtype)+halo
        outputs = []
        for raw in products[:2]:
            filtered = self.source.spectral_leakage(raw[0], dr, sinc_width=leakage)
            filtered = self.source.azimuth_antenna_gain_projection(filtered, new_resolution=filter_resolution,
                                                                  beamwidth=filter_beam)
            # The last sample wraps to the first; no reflection across a local crop.
            data = torch.cat((filtered, filtered[:1]), dim=0).permute(2, 0, 1)[None]
            rows, cols = torch.meshgrid(row, r, indexing="ij")
            query = torch.stack((2*cols/(width-1)-1, 2*rows/(data.shape[-2]-1)-1), -1)[None]
            outputs.append(F.grid_sample(data, query, mode="bilinear", padding_mode="border", align_corners=True)[0, 0])
        power = (outputs[0]+.6*multipath.to(outputs[0])).clamp(0, 1)
        # Preserve the release's physical near-range mask (not a local-bin mask).
        ranges = grid.range_start_m + (torch.arange(grid.num_range_bins, device=device)+.5)*grid.range_resolution_m
        return power * (ranges >= 2.5/self.units), outputs[1]


class ReleasedPreprocessing:
    """Training-only original denoising/periodic fit plus spatial image mapping.

    Mapping replaces the release's driving trajectory with nearest calibrated
    training directions. Validation pixels never fit a multipath model.
    """
    def __init__(self, cache, *, root=REFERENCE_ROOT, intensity_mapping=LINEAR_INTENSITY):
        from scipy.optimize import curve_fit
        from rift.radarsplat_fidelity import OccupancyRecipe, TrainingOccupancy
        if intensity_mapping not in INTENSITY_MAPPINGS:
            raise ValueError(f"unknown RadarSplat intensity mapping {intensity_mapping!r}")
        self.cache, self.intensity_mapping = cache, intensity_mapping
        # Unnormalized FFT magnitude scales with the number of range samples.
        # Preserve the source amplitude threshold when converting its 50 m
        # Navtech crop to our scene crop; retaining 30 for 33 bounded samples
        # would make detection mathematically impossible for every train image.
        self.fft_peak_threshold = 30*int(cache.grid["n_range"])/int(50/.0596)
        self.units = 50/float(cache.grid["scene_extent_m"])
        self.occupancy = TrainingOccupancy(cache, OccupancyRecipe(window_views=11, power_threshold=.10,
                                                                 multipath_fft_peak=self.fft_peak_threshold),
                                           intensity_mapping=intensity_mapping)
        self.functions = source_functions("boreas/data_processing/play_radar_signal.py",
                                         ["FFT", "multipath_modeling"], dict(np=np, torch=torch, curve_fit=curve_fit), root=root)
        self._backgrounds = {}

    def background(self, arrays):
        from rift.radarsplat_fidelity import polar_world_points, sample_polar_power
        center = np.asarray(self.cache.grid["scene_center_m"])
        direction = arrays["sensor_to_world"][:3, 3]-center
        direction = direction/np.linalg.norm(direction)
        donor = min(self.occupancy.train, key=lambda j: (-float(direction @ self.occupancy.directions[j]), j))
        source, _, _ = self.occupancy._donor(donor)
        if donor not in self._backgrounds:
            power = intensity(source["radarsplat_mf_power"].astype(np.float64), self.cache.train_peak_power,
                              self.intensity_mapping)
            dr = self.units*(float(source["range_m"][-1])-float(source["range_m"][0]))/(power.shape[1]-1)
            with np.errstate(divide="ignore", invalid="ignore"):
                _, spectrum, fft, frequencies = self.functions["FFT"](power, range_resolution=dr)
            half = spectrum[:, :power.shape[1]//2]
            ratio = np.divide(half[:, 0], half.sum(1), out=np.zeros(len(half)), where=half.sum(1)>0)
            selected = (half[:, 3:].max(1)>self.fft_peak_threshold) & (ratio>.2)
            result = np.zeros_like(power)
            if selected.any():
                values = self.functions["multipath_modeling"](
                    spectrum, fft, frequencies, selected, power, 3, dr, power.shape[1])
                result[values[0]] = values[7]
            if not np.isfinite(result).all():
                raise ValueError("Released multipath fitting produced nonfinite power")
            self._backgrounds[donor] = result
        points = polar_world_points(arrays)
        values, valid = sample_polar_power(points, source, self._backgrounds[donor])
        shape = (len(arrays["elevation_rad"]), len(arrays["azimuth_rad"]), len(arrays["range_m"]))
        numerator = np.where(valid, values, 0).reshape(shape).sum(0)
        count = valid.reshape(shape).sum(0)
        return np.divide(numerator, count, out=np.zeros_like(numerator), where=count>0).astype(np.float32)
