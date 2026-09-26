#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# Copyright ANSYS. All rights reserved.
#
# PEC sphere FMCW radar simulation - 16T16R array variant
# - Chirp-Sequence FMCW, Range-Doppler, I+Q channels
# - Spherical viewpoint grid: az in [0,45] deg, el in [0,22.5] deg, step 0.25 deg
# - 250,000 Fibonacci-sphere viewpoints
# - Radar-to-target distance: 10 m
#

import os
import sys
import pickle
import numpy as np
import math as m
#import PIL.Image
from collections import defaultdict
from pathlib import Path
from itertools import product as iproduct

# tqdm optional
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):  # noqa
        return x

print("STARTING PEC SPHERE FMCW SIM - 16T16R ARRAY VARIANT")

# ============================================================
# 1. CENTRAL CONFIGURATION
# ============================================================

# --- Derived waveform constants (from screenshot) ---
# Center Frequency:         79 GHz
# Radar Bandwidth:          149.896229 MHz
# Chirp Duration:           12000 us = 12 ms
# A/D Sampling Rate:        0.0025 MHz = 2500 Hz  -> 30 samples/chirp
# # A/D Samples per Chirp:  30
# CPI # of Chirps:          10
# PRF (Chirp Rep. Freq.):   0.026685 kHz = 26.685 Hz
# CPI Duration:             374.740572 ms
# Range Resolution:         1 m
# Range Period:             30 m
# Velocity Resolution:      0.4 m/s
# Velocity Min:            -2 m/s
# Velocity Max:            +2 m/s

# Azimuth:   0 to 45  deg inclusive, step 0.25 deg  -> 181 samples
# Elevation: 0 to 22.5 deg inclusive, step 0.25 deg ->  91 samples
# This variant uses Fibonacci-sphere sampling, so AZ_STEP and EL_STEP are
# retained only as legacy reference values.

AZ_START   =  0.0    # degrees
AZ_END     = 360.0    # degrees
EL_START   =  0.0    # degrees
EL_END     = 180.    # degrees
AZ_STEP    =  0.25   # degrees
EL_STEP    =  0.25   # degrees

#AZ_SAMPLES = int(round((AZ_END - AZ_START) / AZ_STEP)) + 1   # 181
#EL_SAMPLES = int(round((EL_END - EL_START) / EL_STEP)) + 1   # 91
#NUM_FRAMES = AZ_SAMPLES * EL_SAMPLES                           # 16,471
NUM_FRAMES = 2000


LIGHT_SPEED = 299792458.0
RADAR_RADIUS = 10.0
RADAR_FC_HZ = 79e9
PEC_SPHERE_RADIUS_M = 1.0

SIM_CONFIG = {
    # --- Output ---
    "output_base": "pec_sphere_fmcw_16t16r_79ghz_r10m_2k",
    "target": {
        "type": "pec_sphere",
        "radius_m": PEC_SPHERE_RADIUS_M,
        "pos": (0.0, 0.0, 0.0),
        "lat_segments": 48,
        "lon_segments": 96,
    },

    # --- Simulation Timing (dt unused for pose-indexed sim, kept for compat) ---
    "num_frames": NUM_FRAMES,
    "dt": 0.01,

    # --- Scene Selection ---
    # "empty" : no background mesh, only the PEC sphere target
    "SCENE_TYPE": "empty",

    # --- Radar Specifications ---
    # Chirp-Sequence FMCW, I+Q channels, Range-Doppler setup
    "radar": {
        "fc":               RADAR_FC_HZ,   # 79 GHz center frequency
        "bw":               149.896229e6,  # 149.896229 MHz radar bandwidth
        "chirp_duration":   12000e-6,      # 12000 us = 12 ms
        "sampling_rate":    2500.0,        # 0.0025 MHz = 2500 Hz (A/D)
        "num_adc_samples":  30,            # # A/D samples per chirp
        "num_chirps_cpi":   10,            # CPI # of chirps
        "prf":              26.685,        # PRF in Hz (0.026685 kHz)
        "range_res":        1.0,           # meters
        "range_period":     30.0,          # meters
        "vel_res":          0.4,           # m/s
        "vel_min":          -2.0,          # m/s
        "vel_max":           2.0,          # m/s
        "N_tx":             16,
        "N_rx":             16,
        "beam_hpbw_deg":    10.0,          # narrow beam for 1 km standoff
        "channel":          "IQ",          # I+Q channels
        "array": {
            # Device-local +X is the array surface normal and boresight.
            # The per-frame look_at_rotation points +X toward the scene center.
            "normal_axis":       "x",
            "tx_axis":           "y",
            "rx_axis":           "z",
            "element_spacing_lambda": 0.5,
        },
    },

    # --- Solver / Ray Tracing ---
    "solver": {
        "gpu_device_ids": [0],
        "gpu_quota":       1.0,
        "max_refl":        1,
        "max_trans":       0,
        "ray_density":     1.0,
        "min_batches":     2000,
        "max_batches":     5000,
    },

    # --- Ego/radar mounting ---
    "ego": {
        # radar sits at the radar position itself (no offset from platform)
        "radar_offset_pos":   (0.0, 0.0, 0.0),
        # local +X points toward target (origin)
        "radar_forward_axis": "x",
        # target: PEC sphere at world origin
        "look_at_target":     (0.0, 0.0, 0.0),
        "world_up":           (0.0, 0.0, 1.0),
    },

    # --- Save response type ---
    "response_type": "FREQ_PULSE",

    # --- Viewpoint grid (stored for reference / post-processing) ---
    "viewpoint_grid": {
    "type":    "fibonacci_sphere",
    "radius":  RADAR_RADIUS,
    "az_min":  0.0,
    "az_max":  360.0,
    "el_min":  -90.0,   # ← elevation convention: -90 (nadir) to +90 (zenith)
    "el_max":   90.0,
    "n_points": NUM_FRAMES,
    },
}

# ============================================================
# 2. ROBUST LIBRARY SETUP
# ============================================================

THIS_DIR = Path(__file__).resolve().parent

CANDIDATE_LIB_DIRS = [
    THIS_DIR / "lib",
    THIS_DIR / "avx" / "lib",
    THIS_DIR.parent / "avx" / "lib",
]
libDir = next((p for p in CANDIDATE_LIB_DIRS if p.exists()), None)
if libDir is None:
    raise FileNotFoundError(
        f"Could not find RssPy lib folder. Tried: {[str(x) for x in CANDIDATE_LIB_DIRS]}"
    )

CANDIDATE_MODEL_DIRS = [
    THIS_DIR / "models",
    THIS_DIR.parent / "models",
]
MODEL_DIR = next((p for p in CANDIDATE_MODEL_DIRS if p.exists()), None)
if MODEL_DIR is None:
    raise FileNotFoundError(
        f"Could not find models folder. Tried: {[str(x) for x in CANDIDATE_MODEL_DIRS]}"
    )
MODEL_DIR = MODEL_DIR.resolve()
print("Using models dir:", MODEL_DIR)

if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(libDir))

sys.path.insert(0, str(libDir))
os.environ.setdefault("RTR_LICENSE_DIR", str(libDir / "licensingclient"))

import RssPy  # noqa: E402
api = RssPy.RssApi()

# ============================================================
# 3. HELPERS
# ============================================================

def isOK(stat):
    if stat == RssPy.RGpuCallStat.OK:
        return
    elif stat == RssPy.RGpuCallStat.RGPU_WARNING:
        print(f"WARNING: {api.getLastWarnings()}")
        return
    else:
        err = api.getLastError()
        print(f"ERROR: {err}")
        raise RuntimeError(f"RssPy call failed: {err}")


def _normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def _axis_vector(axis_name: str):
    axis = str(axis_name).strip().lower()
    sign = 1.0
    if axis.startswith("+"):
        axis = axis[1:]
    elif axis.startswith("-"):
        sign = -1.0
        axis = axis[1:]

    vectors = {
        "x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0]),
    }
    if axis not in vectors:
        raise ValueError("array axis must be one of: x,y,z,+x,+y,+z,-x,-y,-z")
    return sign * vectors[axis]


def build_linear_array_positions(count: int, axis_name: str, spacing_m: float):
    axis = _axis_vector(axis_name)
    offsets = (np.arange(count, dtype=float) - 0.5 * (count - 1)) * spacing_m
    return offsets[:, None] * axis[None, :]


def look_at_rotation(
    cam_pos_global,
    target_global=(0.0, 0.0, 0.0),
    world_up=(0.0, 0.0, 1.0),
    forward_axis="x",
):
    """
    Build a 3x3 rotation matrix R (local -> global) so the device
    'forward' axis points from cam_pos_global toward target_global.
    """
    cam_pos_global = np.asarray(cam_pos_global, dtype=float)
    target_global  = np.asarray(target_global,  dtype=float)
    up = _normalize(world_up)

    f = _normalize(target_global - cam_pos_global)
    if np.linalg.norm(f) < 1e-12:
        return np.eye(3)

    ax = str(forward_axis).strip().lower()
    sign = 1.0
    if ax.startswith("+"):
        ax = ax[1:]
    elif ax.startswith("-"):
        sign = -1.0
        ax = ax[1:]
    if ax not in ("x", "y", "z"):
        raise ValueError("forward_axis must be one of: x,y,z,+x,+y,+z,-x,-y,-z")

    f = sign * f

    if abs(np.dot(_normalize(f), up)) > 0.99:
        up_try = np.array([0.0, 1.0, 0.0], dtype=float)
        if abs(np.dot(_normalize(f), _normalize(up_try))) > 0.99:
            up_try = np.array([1.0, 0.0, 0.0], dtype=float)
        up = _normalize(up_try)

    r = _normalize(np.cross(up, f))
    if np.linalg.norm(r) < 1e-12:
        up2 = _normalize(np.array([0.0, 1.0, 0.0], dtype=float))
        r = _normalize(np.cross(up2, f))
        if np.linalg.norm(r) < 1e-12:
            return np.eye(3)
    u = np.cross(f, r)

    if ax == "x":
        R = np.stack([f, r, u], axis=1)
    elif ax == "y":
        R = np.stack([r, f, u], axis=1)
    else:
        R = np.stack([r, u, f], axis=1)

    return R


class CoordSys:
    def __init__(self, hNode=None, hElem=None):
        if hNode is None:
            hNode = RssPy.SceneNode()
            isOK(api.addSceneNode(hNode))
        self.hNode = hNode
        self.hElem = hElem
        self.rot = np.eye(3)
        self.pos = np.zeros(3)
        self.lin = np.zeros(3)
        self.ang = np.zeros(3)
        if hElem is not None:
            isOK(api.setSceneElement(self.hNode, self.hElem))

    def update(self, time):
        newPos = np.asarray(self.pos) + time * np.asarray(self.lin)
        isOK(api.setCoordSysInGlobal(self.hNode, self.rot, newPos, self.lin, self.ang))


def loadMesh(filename):
    mesh = api.loadTriangleMesh(str(filename))
    hMesh = RssPy.SceneElement()
    isOK(api.addSceneElement(hMesh))
    isOK(api.setTriangles(hMesh, mesh))
    return (hMesh, mesh)


def write_pec_sphere_stl(filename, radius_m: float, lat_segments: int = 48,
                         lon_segments: int = 96):
    import struct

    radius_m = float(radius_m)
    lat_segments = int(lat_segments)
    lon_segments = int(lon_segments)
    if radius_m <= 0:
        raise ValueError("sphere radius must be positive")
    if lat_segments < 4 or lon_segments < 8:
        raise ValueError("sphere mesh needs at least 4 latitude and 8 longitude segments")

    def vertex(theta, phi):
        return np.array([
            radius_m * np.sin(theta) * np.cos(phi),
            radius_m * np.sin(theta) * np.sin(phi),
            radius_m * np.cos(theta),
        ], dtype=np.float32)

    def oriented_triangle(a, b, c):
        normal = np.cross(b - a, c - a)
        center = (a + b + c) / 3.0
        if np.dot(normal, center) < 0:
            b, c = c, b
            normal = np.cross(b - a, c - a)
        n = np.linalg.norm(normal)
        if n > 0:
            normal = normal / n
        return normal.astype(np.float32), a, b, c

    triangles = []
    for i_lat in range(lat_segments):
        theta0 = np.pi * i_lat / lat_segments
        theta1 = np.pi * (i_lat + 1) / lat_segments
        for i_lon in range(lon_segments):
            phi0 = 2.0 * np.pi * i_lon / lon_segments
            phi1 = 2.0 * np.pi * (i_lon + 1) / lon_segments

            p00 = vertex(theta0, phi0)
            p01 = vertex(theta0, phi1)
            p10 = vertex(theta1, phi0)
            p11 = vertex(theta1, phi1)

            if i_lat == 0:
                triangles.append(oriented_triangle(p00, p10, p11))
            elif i_lat == lat_segments - 1:
                triangles.append(oriented_triangle(p00, p10, p01))
            else:
                triangles.append(oriented_triangle(p00, p10, p11))
                triangles.append(oriented_triangle(p00, p11, p01))

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    header = b"PEC sphere generated by sim_pec_sphere_fmcw_16t16r_79ghz_2k.py"
    header = header[:80].ljust(80, b"\x00")

    with open(filename, "wb") as f:
        f.write(header)
        f.write(struct.pack("<I", len(triangles)))
        for normal, a, b, c in triangles:
            f.write(struct.pack("<3f", *normal))
            f.write(struct.pack("<3f", *a))
            f.write(struct.pack("<3f", *b))
            f.write(struct.pack("<3f", *c))
            f.write(b"\x00\x00")

    return filename

# ============================================================
# 4. VIEWPOINT SAMPLING
# ============================================================

def build_viewpoint_grid(config: dict):
    """
    Generate all (az, el) pairs on a uniform angular grid, then convert to
    Cartesian positions on the sphere of radius R.

    Convention:
      - Azimuth (az): angle in the XY plane from +X axis, measured toward +Y
      - Elevation (el): angle above the XY plane (0 = horizontal, 90 = zenith)
      - Positive elevation: radar is above the target

    Returns:
      positions : (N, 3) float array  — radar positions in world coords
      angles    : (N, 2) float array  — (az_deg, el_deg) for each row
    """
    vg = config["viewpoint_grid"]
    R       = float(vg["radius"])
    az_vals = np.linspace(vg["az_start"], vg["az_end"], vg["az_samples"])  # degrees
    el_vals = np.linspace(vg["el_start"], vg["el_end"], vg["el_samples"])  # degrees

    positions = []
    angles    = []

    for el_deg in el_vals:
        for az_deg in az_vals:
            az_r = np.deg2rad(az_deg)
            el_r = np.deg2rad(el_deg)

            x = R * np.cos(el_r) * np.cos(az_r)
            y = R * np.cos(el_r) * np.sin(az_r)
            z = R * np.sin(el_r)

            positions.append([x, y, z])
            angles.append([az_deg, el_deg])

    return np.array(positions, dtype=float), np.array(angles, dtype=float)

def build_viewpoint_grid_FIB(config: dict):
    vg = config["viewpoint_grid"]
    R         = float(vg["radius"])
    n_target  = int(vg["n_points"])
    az_min    = float(vg.get("az_min",   0.0))
    az_max    = float(vg.get("az_max", 360.0))
    el_min    = float(vg.get("el_min", -90.0))   # elevation: -90 to +90
    el_max    = float(vg.get("el_max",  90.0))

    full_sphere = (
        abs(az_max - az_min - 360.0) < 1e-6 and
        abs(el_min - (-90.0))        < 1e-6 and
        abs(el_max -   90.0)         < 1e-6
    )

    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0

    if full_sphere:
        # --- Direct generation: no filtering needed ---
        indices = np.arange(n_target)
        sin_el  = 2.0 * indices / (n_target - 1) - 1.0   # uniform in [-1, 1]
        el_deg  = np.rad2deg(np.arcsin(sin_el))
        az_deg  = (360.0 * indices / golden_ratio) % 360.0

    else:
        # --- Partial patch: oversample full sphere, then filter ---
        # Solid angle fraction: dΩ = daz * d(sin_el)
        az_frac = (az_max - az_min) / 360.0
        el_frac = (np.sin(np.deg2rad(el_max)) - np.sin(np.deg2rad(el_min))) / 2.0
        patch_fraction = az_frac * el_frac
        if patch_fraction <= 0:
            raise ValueError(
                f"Patch solid angle is zero — check az/el bounds. "
                f"Got az=[{az_min},{az_max}], el=[{el_min},{el_max}]"
            )

        oversample = 6
        N_full = int(np.ceil(n_target / patch_fraction * oversample))
        indices = np.arange(N_full)
        sin_el_all = 2.0 * indices / (N_full - 1) - 1.0
        el_deg_all = np.rad2deg(np.arcsin(sin_el_all))
        az_deg_all = (360.0 * indices / golden_ratio) % 360.0

        mask   = (
            (az_deg_all >= az_min) & (az_deg_all <= az_max) &
            (el_deg_all >= el_min) & (el_deg_all <= el_max)
        )
        el_deg = el_deg_all[mask]
        az_deg = az_deg_all[mask]

        actual = len(el_deg)
        print(f"[Fibonacci grid] {actual} points after filtering "
              f"(target={n_target}, N_full={N_full}, patch_fraction={patch_fraction:.4f})")
        if actual == 0:
            raise RuntimeError("No points in patch — increase oversample or widen bounds.")
        if actual < int(n_target * 0.8):
            print(f"  WARNING: got {actual} < 80% of target {n_target}. "
                  f"Increase oversample (currently {oversample}).")

    # --- Convert to Cartesian ---
    az_r = np.deg2rad(az_deg)
    el_r = np.deg2rad(el_deg)
    x = R * np.cos(el_r) * np.cos(az_r)
    y = R * np.cos(el_r) * np.sin(az_r)
    z = R * np.sin(el_r)

    positions = np.stack([x, y, z], axis=1)
    angles    = np.stack([az_deg, el_deg], axis=1)

    print(f"[Fibonacci grid] Final: {len(positions)} viewpoints on R={R} m sphere")
    return positions, angles

# ============================================================
# 5. SIMULATOR
# ============================================================

class SimulatorWrapper:
    def __init__(self, config: dict, data_folder: str, scene_type: str = None):
        self.config = config

        self.meshs  = defaultdict(list)
        self.tmeshs = defaultdict(list)
        self.csList = defaultdict(list)

        self.rxAntennas = []
        self.txAntennas = []
        self.rx_positions_local = np.zeros((0, 3))
        self.tx_positions_local = np.zeros((0, 3))
        self.N_rx = 0
        self.N_tx = 0

        self.hMode   = None
        self.hDevice = None
        self.radar_offset_pos    = np.zeros(3)
        self.radar_forward_axis  = "x"

        self.data_folder = str(data_folder)
        os.makedirs(self.data_folder, exist_ok=True)

        if scene_type is None:
            scene_type = config.get("SCENE_TYPE", "empty")
        self.scene_type = scene_type

        # No background, only the configured target
        if scene_type not in ("empty", "none"):
            raise NotImplementedError(
                f"This script only supports scene_type='empty'. Got: {scene_type!r}"
            )
        print("[scene] empty: no background mesh")

    # ---------------------------
    # Objects
    # ---------------------------

    def add_pec_sphere(self, pos=(0.0, 0.0, 0.0), radius_m: float = PEC_SPHERE_RADIUS_M,
                       lat_segments: int = 48, lon_segments: int = 96):
        """
        Create a generated sphere triangle mesh and load it as an uncoated
        scene object. In this RssPy setup, uncoated object surfaces are used
        for the PEC target behavior.
        """
        sphere_stl = Path(self.data_folder) / f"_pec_sphere_r{float(radius_m):g}m.stl"
        sphere_stl = write_pec_sphere_stl(
            sphere_stl,
            radius_m=radius_m,
            lat_segments=lat_segments,
            lon_segments=lon_segments,
        )
        (hMesh, mesh) = loadMesh(sphere_stl)

        cs = CoordSys(None, hMesh)
        cs.pos = np.asarray(pos, dtype=float)
        cs.lin = np.zeros(3)
        cs.rot = np.eye(3)

        self.meshs["pec_sphere"].append(hMesh)
        self.tmeshs["pec_sphere"].append(mesh)
        self.csList["pec_sphere"].append(cs)
        print(
            f"[target] PEC sphere added at {tuple(cs.pos)} with radius={float(radius_m):.6f} m "
            f"({int(lat_segments)}x{int(lon_segments)} mesh)"
        )

    # ---------------------------
    # Ego platform + radar device
    # ---------------------------

    def add_ego_radar(self, pos):
        """
        Add a minimal 'point' radar platform (no mesh) at the given position.
        The radar device sits directly at the platform with no offset.
        """
        ego_cfg = self.config.get("ego", {})
        offset_pos = np.asarray(ego_cfg.get("radar_offset_pos", (0.0, 0.0, 0.0)), dtype=float)
        self.radar_offset_pos   = offset_pos
        self.radar_forward_axis = str(ego_cfg.get("radar_forward_axis", "x")).lower()

        hPlatform = RssPy.RadarPlatform()
        isOK(api.addRadarPlatform(hPlatform))

        # Platform coord sys — no mesh
        platformCS = CoordSys(hPlatform, None)
        platformCS.pos = np.asarray(pos, dtype=float)
        platformCS.lin = np.zeros(3)
        platformCS.rot = np.eye(3)
        self.csList["ego"].append(platformCS)

        hDevice = RssPy.RadarDevice()
        isOK(api.addRadarDevice(hDevice, hPlatform))

        radarCS = CoordSys(hDevice)
        radarCS.rot = np.eye(3)
        isOK(api.setCoordSysInParent(hDevice, radarCS.rot, offset_pos,
                                     radarCS.lin, radarCS.ang))

        self.hDevice = hDevice
        self.radarCS = radarCS

    # ---------------------------
    # Radar sensor configuration
    # ---------------------------

    def configure_radar_sensor(self):
        rc = self.config["radar"]
        sc = self.config["solver"]

        centerFreq = float(rc["fc"])
        bandwidth  = float(rc["bw"])           # 149.896229 MHz
        N_tx       = int(rc["N_tx"])
        N_rx       = int(rc["N_rx"])

        self.N_tx = N_tx
        self.N_rx = N_rx

        hpbw = float(rc.get("beam_hpbw_deg", 10.0))

        # --- Antennas ---
        hTxs = []
        hRxs = []
        for _ in range(N_tx):
            hTx = RssPy.RadarAntenna()
            isOK(api.addRadarAntennaParametricBeam(
                hTx, self.hDevice,
                RssPy.AntennaPolarization.VERT, hpbw, hpbw, 1.0
            ))
            hTxs.append(hTx)

        for _ in range(N_rx):
            hRx = RssPy.RadarAntenna()
            isOK(api.addRadarAntennaParametricBeam(
                hRx, self.hDevice,
                RssPy.AntennaPolarization.VERT, hpbw, hpbw, 1.0
            ))
            hRxs.append(hRx)

        # --- Mode ---
        hMode = RssPy.RadarMode()
        isOK(api.addRadarMode(hMode, self.hDevice))
        isOK(api.setRadarModeActive(hMode, True))

        for hTx in hTxs:
            isOK(api.addTxAntenna(hMode, hTx))
        for hRx in hRxs:
            isOK(api.addRxAntenna(hMode, hRx))

        print(f"Radar Configured: {api.numTxAntennas(hMode)} TX, {api.numRxAntennas(hMode)} RX")

        # --- Antenna placement: orthogonal TX/RX line arrays in the local YZ plane ---
        array_cfg = rc.get("array", {})
        tx_axis = array_cfg.get("tx_axis", "y")
        rx_axis = array_cfg.get("rx_axis", "z")
        normal_axis = array_cfg.get("normal_axis", "x")
        wavelength_m = LIGHT_SPEED / centerFreq
        spacing_lambda = float(array_cfg.get("element_spacing_lambda", 0.5))
        spacing_m = spacing_lambda * wavelength_m

        tx_axis_vec = _axis_vector(tx_axis)
        rx_axis_vec = _axis_vector(rx_axis)
        normal_vec = _axis_vector(normal_axis)
        surface_normal = _normalize(np.cross(tx_axis_vec, rx_axis_vec))
        if np.dot(surface_normal, normal_vec) < 0.999:
            raise ValueError(
                "TX/RX axes must define the configured array normal. "
                f"Got tx_axis={tx_axis!r}, rx_axis={rx_axis!r}, "
                f"normal_axis={normal_axis!r}."
            )

        tx_positions = build_linear_array_positions(N_tx, tx_axis, spacing_m)
        rx_positions = build_linear_array_positions(N_rx, rx_axis, spacing_m)

        ant_rot = np.eye(3)
        self.txAntennas = api.txAntennas(hMode)[1]
        self.rxAntennas = api.rxAntennas(hMode)[1]

        for hTx, ant_pos in zip(self.txAntennas, tx_positions):
            isOK(api.setCoordSysInParent(hTx, ant_rot,
                                         ant_pos, [0, 0, 0], [0, 0, 0]))
        for hRx, ant_pos in zip(self.rxAntennas, rx_positions):
            isOK(api.setCoordSysInParent(hRx, ant_rot,
                                         ant_pos, [0, 0, 0], [0, 0, 0]))

        self.tx_positions_local = tx_positions
        self.rx_positions_local = rx_positions

        tx_aperture = spacing_m * (N_tx - 1)
        rx_aperture = spacing_m * (N_rx - 1)
        print(
            "Radar array geometry: "
            f"TX {N_tx} along local {tx_axis} ({tx_aperture:.6f} m aperture), "
            f"RX {N_rx} along local {rx_axis} ({rx_aperture:.6f} m aperture), "
            f"spacing={spacing_m:.9f} m ({spacing_lambda:.3f} lambda), "
            f"normal=local {normal_axis}"
        )

        # --- Waveform: Chirp-Sequence FMCW, Range-Doppler ---
        # Parameters from screenshot:
        #   Chirp duration    = 12000 us  (pulseInterval = chirp rep period = 1/PRF)
        #   # chirps per CPI  = 10
        #   # ADC samples     = 30        (num_freq_samples)
        #   Bandwidth         = 149.896229 MHz
        #   Center frequency  = configured radar fc
        #
        # pulseInterval = 1 / PRF = 1 / 26.685 Hz = ~37.474 ms
        # This matches CPI duration / num_chirps = 374.740572 ms / 10 = 37.474 ms

        num_chirps       = int(rc["num_chirps_cpi"])       # 10
        num_adc_samples  = int(rc["num_adc_samples"])      # 30
        prf_hz           = float(rc["prf"])                # 26.685 Hz
        pulse_interval   = 1.0 / prf_hz                   # ~37.474 ms
        tx_multiplex     = RssPy.TxMultiplex.INTERLEAVED
        num_pulse_cpi    = N_tx * num_chirps

        isOK(api.setPulseDopplerWaveformSysSpecs(
            hMode,
            centerFreq,       # center frequency (Hz)
            bandwidth,        # bandwidth (Hz)
            num_adc_samples,  # # frequency/ADC samples per chirp
            pulse_interval,   # chirp repetition interval (s)
            num_pulse_cpi,    # total pulses per CPI
            tx_multiplex,
        ))

        # --- Solver ---
        print("GPUs:", api.listGPUs())
        dev_ids   = list(sc.get("gpu_device_ids", [0]))
        dev_quota = float(sc.get("gpu_quota", 1.0))
        isOK(api.setGPUDevices(dev_ids, [dev_quota] * len(dev_ids)))

        min_batches = int(sc.get("min_batches", 0))
        max_batches = int(sc.get("max_batches", 100))
        if min_batches > 0 and hasattr(api, "setMinNumRayBatches"):
            isOK(api.setMinNumRayBatches(min_batches))
        isOK(api.autoConfigureSimulation(max_batches))

        isOK(api.setMaxNumRefl(int(sc.get("max_refl", 1))))
        isOK(api.setMaxNumTrans(int(sc.get("max_trans", 0))))
        ray_density = float(sc.get("ray_density", 0.01))
        isOK(api.setTargetRayDensity(ray_density))
        print(
            f"Solver ray settings: density={ray_density:g}, "
            f"min_batches={min_batches}, max_batches={max_batches}"
        )

        # --- Radar camera ---
        #isOK(api.initRadarCamera(
            #RssPy.CameraProjection.FISHEYE,
            #RssPy.CameraColorMode.COATING,
            #512, 512, 255,
            #True,
            #1000.0
        #))
        #isOK(api.activateRadarCamera())

        self.hMode = hMode

        if not api.isReady():
            print("WARNING: RSS is not ready:\n", api.getLastWarnings())

    # ---------------------------
    # Logging / I/O
    # ---------------------------

    def get_min_max(self, tmesh):
        v = np.array(tmesh.vertices)
        return np.min(v, axis=0), np.max(v, axis=0)

    def write_metadata(self, frame_num: int, az_deg: float, el_deg: float,
                        radar_pos: np.ndarray):
        meta = {
            "frame":     frame_num,
            "az_deg":    az_deg,
            "el_deg":    el_deg,
            "radar_pos": radar_pos,
        }

        # TX positions
        tx_positions = []
        for tx in self.txAntennas:
            tx_positions.append(api.coordSysInGlobal(tx)[2])
        meta["tx_pos"] = np.array(tx_positions)
        meta["tx_pos_local"] = np.array(self.tx_positions_local)

        # RX positions
        rx_positions = []
        for rx in self.rxAntennas:
            rx_positions.append(api.coordSysInGlobal(rx)[2])
        meta["rx_pos"] = np.array(rx_positions)
        meta["rx_pos_local"] = np.array(self.rx_positions_local)

        # Device position
        if self.hDevice is not None:
            meta["device_pos"] = api.coordSysInGlobal(self.hDevice)[2]

        # Object AABBs
        for obj in self.tmeshs:
            for cs, tmesh in zip(self.csList[obj], self.tmeshs[obj]):
                lmin, lmax = self.get_min_max(tmesh)
                cs_pos = api.coordSysInGlobal(cs.hNode)[2]
                meta.setdefault(obj, []).append((lmin + cs_pos, lmax + cs_pos))

        with open(f"{self.data_folder}/meta{frame_num:05d}.pkl", "wb") as f:
            pickle.dump(meta, f)

    # ---------------------------
    # Main loop
    # ---------------------------

    def run_sim(self):
        ego_conf = self.config.get("ego", {})
        target   = np.asarray(ego_conf.get("look_at_target", (0.0, 0.0, 0.0)), dtype=float)
        world_up = ego_conf.get("world_up", (0.0, 0.0, 1.0))

        positions, angles = build_viewpoint_grid_FIB(self.config)
        num_frames = len(positions)

        # Replace the old prints that reference AZ_SAMPLES / EL_SAMPLES with:
        print(f"\nStarting PEC Sphere FMCW Sim")
        print(f"  Viewpoints : {num_frames} (Fibonacci sphere)")
        print(f"  Az range   : {AZ_START}° to {AZ_END}°")
        print(f"  El range   : {EL_START}° to {EL_END}°")
        print(f"  Radius     : {RADAR_RADIUS} m")
        print(f"  Output     : {self.data_folder}\n")

        # Save the full viewpoint table once
        np.save(f"{self.data_folder}/viewpoint_positions.npy", positions)
        np.save(f"{self.data_folder}/viewpoint_angles.npy",    angles)

        for iFrame in tqdm(range(num_frames), desc="Simulating", unit="frame"):
            radar_pos = positions[iFrame]
            az_deg, el_deg = angles[iFrame]

            # --- Move radar platform to current viewpoint ---
            self.csList["ego"][0].pos = radar_pos
            self.csList["ego"][0].lin = np.zeros(3)

            # Update all coord systems (t=0 since nothing is moving)
            for obj_type in self.csList:
                for cs in self.csList[obj_type]:
                    cs.update(0.0)

            # --- Orient radar to look at target (origin) ---
            if self.hDevice is not None:
                R_global = look_at_rotation(
                    cam_pos_global=radar_pos,
                    target_global=target,
                    world_up=world_up,
                    forward_axis=self.radar_forward_axis,
                )
                # Platform has identity rotation, so R_parent = R_global
                isOK(api.setCoordSysInParent(
                    self.hDevice,
                    R_global,
                    self.radar_offset_pos,
                    np.zeros(3),
                    np.zeros(3),
                ))

            # --- Write metadata ---
            self.write_metadata(iFrame, az_deg, el_deg, radar_pos)

            # --- Run ray trace ---
            stat = api.computeResponseSync()
            if stat == RssPy.RGpuCallStat.RGPU_WARNING:
                tqdm.write(f"[RSS WARNING] {api.getLastWarnings()}")
            isOK(stat)

            # --- Retrieve and save response ---
            (ret, response) = api.retrieveResponse(self.hMode, RssPy.ResponseType.FREQ_PULSE)
            isOK(ret)

            arr = np.array(response)
            if not np.isfinite(arr).all():
                tqdm.write(f"\n[CRITICAL FAILURE] Frame {iFrame} "
                           f"(az={az_deg:.2f}°, el={el_deg:.2f}°): NaNs in response.")
                break

            # Save as [num_adc_samples x num_chirps] complex array
            # Shape depends on RssPy's FREQ_PULSE layout: (N_tx, N_rx, N_freq, N_pulses)
            np.save(f"{self.data_folder}/frame{iFrame:05d}.npy", arr)

            # --- Radar camera image ---
            #(ret, radarCamera, _, _) = api.retrieveRadarCameraImage(self.hMode)
            #if ret == RssPy.RGpuCallStat.OK:
                #im = PIL.Image.frombytes("RGB", radarCamera.shape[0:2], radarCamera)
                #im.save(f"{self.data_folder}/camera{iFrame:05d}.png")

        print(f"\nSimulation complete. {num_frames} frames saved to '{self.data_folder}'.")

# ============================================================
# 6. MAIN
# ============================================================

def main():
    try:
        print(api.copyright())
        print(api.version(True))
    except Exception:
        pass

    if hasattr(api, "selectApiLicenseMode") and hasattr(RssPy, "ApiLicenseMode"):
        try:
            isOK(api.selectApiLicenseMode(RssPy.ApiLicenseMode.PERCEIVE_EM))
            print("Selected license mode: PERCEIVE_EM")
        except Exception as e:
            print(f"License mode selection not applied (continuing): {e}")

    out_base   = THIS_DIR / SIM_CONFIG.get("output_base", "pec_sphere_fmcw")
    scene_type = SIM_CONFIG.get("SCENE_TYPE",  "empty")

    sim = SimulatorWrapper(
        config=SIM_CONFIG,
        data_folder=out_base,
        scene_type=scene_type,
    )

    # --- Place PEC sphere at world origin ---
    target_cfg = SIM_CONFIG.get("target", {})
    sim.add_pec_sphere(
        pos=target_cfg.get("pos", (0.0, 0.0, 0.0)),
        radius_m=float(target_cfg.get("radius_m", PEC_SPHERE_RADIUS_M)),
        lat_segments=int(target_cfg.get("lat_segments", 48)),
        lon_segments=int(target_cfg.get("lon_segments", 96)),
    )

    # --- Place radar at first viewpoint (will be overridden each frame) ---
    init_positions, _ = build_viewpoint_grid_FIB(SIM_CONFIG)
    init_pos = init_positions[0]
    print(f"Initializing radar platform at: {init_pos}")
    sim.add_ego_radar(pos=init_pos)

    # --- Configure radar sensor ---
    sim.configure_radar_sensor()

    # --- Run ---
    sim.run_sim()


if __name__ == "__main__":
    main()
