"""Radar array / physical constants shared across the RIFT pipeline.

Values match the virtual MIMO array used to generate the AEDT training data
(16x16 Tx/Rx array, 100 GHz center frequency, 10 GHz bandwidth).
"""
import torch

cc = 299792458.0  # speed of light, m/s

num_tx = num_rx = 16
bw = 10e9
fc = 100e9
nf = 1000
f_hi = 105e9
spacing = cc / (f_hi * 2)

# Radar-to-scene distance for the AEDT data (meters)
arr_dist = 10.0

# Scene voxel grid
pos_encoding_degree = 10
grid_dimension = 3
fp_granularity = 24
extent = 3

# Historical Cartesian/Stolt NUFFT prototype tunables, retained for compatibility.
# rift/nufft_forward_operator.py was retired; these are NOT the settings of the
# current rift/range_operator.py (pure torch Gaussian gridding and torch.fft,
# no torchkbnufft dependency). The historical prototype used torchkbnufft:
# See memory/manuscript notes on the Muppala-derived wavenumber-domain
# design: scene_fft3 zero-pads the (regular) scene grid before the FFT;
# the dispersion grid is the dense per-viewpoint (kx,ky) sampling used for
# the Type-2 NUFFT resample; kernel_width is torchkbnufft's `numpoints`
# (Kaiser-Bessel interpolation width, accuracy/cost knob). guard_band and
# dispersion_grid_size must be scaled up TOGETHER (empirically: widening
# guard_band alone at fixed grid size makes accuracy WORSE, via coarser
# dkx/undersampling -- see project memory) -- 128/3.0 is a validated-better
# pairing than the original 64/1.5, though residual error at real training
# scale (extent=3m) is still an open problem, see forward_operator_nufft.
nufft_grid_oversample = 2.0
nufft_dispersion_grid_size = 384
nufft_kernel_width = 6
nufft_guard_band = 3.0


def get_freqs(nf=nf, fc=fc, bw=bw):
    return torch.linspace(fc - bw / 2, fc + bw / 2, nf)
