"""GOTCHA calibration array: the dataset's own reference reflectors measured in its TRAIN phase history.

The GOTCHA volumetric release ships no transmit power or radiometric constant: each
MAT file holds only fp, freq, x/y/z, r0, th, phi and the autofocus solution. Its
reference document does tabulate a calibration array (Casteel et al., "A Challenge
Problem for 2D/3D Imaging of Targets from a Volumetric Data Set in an Urban
Environment", Proc. SPIE 6568, 65680D, 2007, Table 1). This module measures a
system constant from those reflectors.

What is the dataset's and what is ours: the reflector types, sizes, positions and
headings are the dataset's (Table 1). The closed-form radar cross sections (triangular
trihedral, 4 pi a^4 / (3 lambda^2) at the band centre; a square trihedral would be
9.5 dB larger) and the resulting constant are our computation.

Measurement: for each trihedral, a coherent sub-aperture image over the TRAIN sectors
within +-6 degrees of its boresight (compass heading h -> platform azimuth 90 - h), with
the co-pol autofocus applied once by the native reader and the native kernel
exp(-i 4 pi f (|x - a| - r_ref) / c). A coarse +-1.5 m window at 10 cm is refined at 1 cm
around its peak. For a point target the coherent average returns its per-sample
amplitude s, so K = s (Rt + Rr)^2 / sqrt(sigma) expresses the data in the units of
GeRaF's released amplitude law (sigma * specular / (Rt + Rr)^2): the data divided by K
equal sqrt(RCS) / (Rt + Rr)^2.
"""
from __future__ import annotations

import math

import numpy as np

from .gotcha_dataset import C

SOURCE = ('Casteel, Gorham, Minardi, Scarborough, Naidu, Majumder, "A Challenge Problem for 2D/3D '
          'Imaging of Targets from a Volumetric Data Set in an Urban Environment", Proc. SPIE 6568, '
          '65680D (2007), Table 1')
INCH = 0.0254
# Table 1 trihedrals: id -> (edge length m, X, Y, Z m, compass heading deg). Dihedrals and
# the tophat are omitted: their response depends on orientation/polarization or is
# not tabulated.
TRIHEDRALS = {
    '15TR-01': (15 * INCH, -32.14, 42.54, -0.53, 180), '15TR-03': (15 * INCH, -28.09, 38.67, -0.42, 90),
    '15TR-04': (15 * INCH, -13.86, 37.70, -0.05, 0), '15TR-05': (15 * INCH, -24.39, 32.96, -0.33, 270),
    '15TR-06': (15 * INCH, -32.50, 33.41, -0.57, 0), '15TR-07': (15 * INCH, -5.12, 22.98, -0.05, 270),
    '27TR-01': (27 * INCH, -7.51, 51.47, -0.09, 180),
}
BORESIGHT_HALF_WIDTH_DEG = 6.0


def triangular_trihedral_rcs(edge_m, wavelength_m):
    """Peak radar cross section of a triangular trihedral corner reflector."""
    return 4 * math.pi * edge_m ** 4 / (3 * wavelength_m ** 2)


def boresight_azimuth(heading_deg):
    """Platform azimuth (math convention, from +x counter-clockwise) facing a compass heading."""
    return (90.0 - heading_deg) % 360.0


def boresight_sectors(sectors, heading_deg, half_width=BORESIGHT_HALF_WIDTH_DEG):
    az = boresight_azimuth(heading_deg)
    return sorted(s for s in sectors if min(abs(s - 0.5 - az), 360 - abs(s - 0.5 - az)) <= half_width)


def coherent_image(observations, points):
    """Mean over pulses of the per-pulse matched response at ``points`` (complex, one per point)."""
    total = np.zeros(len(points), dtype=np.complex128)
    for obs in observations:
        delay = np.linalg.norm(points - obs.position_m, axis=1) - obs.reference_range_m
        kernel = np.exp((4j * math.pi / C) * np.outer(delay, obs.frequencies_hz))
        total += kernel @ obs.response / len(obs.frequencies_hz)
    return total / len(observations)


def plane_grid(center, half_extent, step):
    axis = np.arange(-half_extent, half_extent + step / 2, step)
    gx, gy = np.meshgrid(center[0] + axis, center[1] + axis, indexing='ij')
    return np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, center[2])], 1)


def measure_trihedral(observations, position, *, coarse=(1.5, 0.10), fine=(0.12, 0.01), background_offset=5.0):
    """Peak per-sample amplitude of one reflector, its offset from Table 1, and a background reference."""
    position = np.asarray(position, dtype=np.float64)
    grid = plane_grid(position, *coarse)
    image = np.abs(coherent_image(observations, grid))
    peak = grid[image.argmax()]
    fine_grid = plane_grid(peak, *fine)
    fine_image = np.abs(coherent_image(observations, fine_grid))
    best = fine_grid[fine_image.argmax()]
    background = np.abs(coherent_image(observations, plane_grid(position + [background_offset, background_offset, 0], *coarse))).max()
    ranges = np.array([np.linalg.norm(best - o.position_m) for o in observations])
    return dict(amplitude=float(fine_image.max()), offset_m=[float(v) for v in best - position],
                background_amplitude=float(background), mean_range_m=float(ranges.mean()),
                pulses=len(observations))


def system_constant(amplitude, mean_range_m, rcs_m2):
    """K such that data / K = sqrt(RCS) / (Rt + Rr)^2 (monostatic: Rt + Rr = 2 R)."""
    return amplitude * (2 * mean_range_m) ** 2 / math.sqrt(rcs_m2)


# Measured 2026-09-22 by scripts/gotcha_calibration_array.py (CPU job 2156302; artifact
# RIFT_pvc_runs/gotcha_calibration/gotcha_calibration_hh.json). HH, all eight passes, the
# production split's TRAIN sectors only. Median over 48 detections (six 15-inch trihedrals x
# 8 passes, each 18-190x background): relative MAD 11%, per-pass medians 6950-8855, 27-inch
# cross-check 9501 (+1.5 dB). The Table 1 positions sit a consistent (+0.06, -0.49) m off
# in the delivered data frame, a documentation/data discrepancy recorded as found.
# This is a PRESENTATION constant, not a training one: no fit uses it. GOTCHA results
# divided by it read as sqrt(RCS) per resolution cell. Training stays in each method's
# native or TRAIN-normalized units; GeRaF keeps trans_power = 1 because calibrated units
# would put its gradients (~5e-13) below AdamW's eps.
GOTCHA_HH_SYSTEM_CONSTANT = 7987.689272473237
GOTCHA_TABLE1_OFFSET_M = (0.06, -0.49, 0.0)
