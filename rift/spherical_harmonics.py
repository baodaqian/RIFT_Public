"""Real spherical harmonics (degree 0-10), evaluated on a single viewing
direction given as (theta, phi) in the same convention as
forward_operator.get_array_pos (theta = polar angle from +z, phi =
azimuth). Same normalization convention (and, for degree <=2, the exact
same numeric constants) as Plenoxels (svox2) and most NeRF-style SH-color
implementations, so a degree-0 basis is just the constant C0 -- see
SHVoxelGridScene in rift/sparse_scene.py, where that fact is used as a
regression check (degree-0 must reduce to the isotropic VoxelGridScene up
to a fixed scale).

Basis functions are generated degree-major, m ordered -l..l within each
degree block (matching the original hand-written degree<=2 table this
module replaced): index 0 = Y_0^0, indices 1-3 = Y_1^{-1,0,1}, indices 4-8
= Y_2^{-2,-1,0,1,2}, etc. num_sh_basis(l) - num_sh_basis(l-1) == 2l+1 new
basis functions are appended at each degree -- SHVoxelGridScene relies on
this to map a basis index back to the degree it belongs to.

Associated Legendre polynomials P_l^m(cos theta) are evaluated via the
standard stable three-term recurrence (Sloan, "Stupid Spherical Harmonics
Tricks", 2008 -- the same recurrence used by most real-time SH-lighting
and SH-scene-representation code, and by high-degree spherical-harmonic-
transform libraries like SHTns/healpy). Numerically stable to at least a
few thousand degree in float64 (the only failure mode at very high m is
graceful underflow of P_m^m towards zero, not instability); the practical
ceiling here (MAX_SUPPORTED_DEGREE=2000) exists to bound cost (this
module's per-degree loop is O(L^2) in pure Python/torch, not to reflect a
numerical limit), not accuracy. rift/sparse_scene.py's SHVoxelGridScene
only ever asks for degree <=10 (see its own CLI cap in train.py); the
higher range exists for rift/swe_operator.py, whose spherical-wave
expansions can genuinely need L in the hundreds to thousands (L must
grow with K*r -- see that module's docstring).
"""
import math

import torch

MAX_SUPPORTED_DEGREE = 2000


def num_sh_basis(degree):
    if not (0 <= degree <= MAX_SUPPORTED_DEGREE):
        raise ValueError(f"degree must be 0..{MAX_SUPPORTED_DEGREE}, got {degree}")
    return (degree + 1) ** 2


def physical_max_degree(pitch, f_max_hz, c=299792458.0, safety=1.0, cap=10):
    """The angular bandwidth a SINGLE voxel can physically carry, in SH degree.

    A scatterer confined to radius a radiates a field band-limited to
    L ~ k_eff * a. Here the pattern is the MONOSTATIC response as the radar
    direction moves, so the relevant wavenumber is the round-trip 2k
    (the phase is exp(i*2k*u_hat . x) for a monostatic look direction u_hat),
    and a is the voxel's half-diagonal, a = sqrt(3)/2 * pitch:

        L_phys = ceil( safety * 2 * (2*pi*f_max/c) * (sqrt(3)/2) * pitch )

    Above L_phys a voxel's SH coefficients cannot describe any physical
    scattering pattern -- they only add free parameters that fit
    training views and do not generalize. Note the DIRECTION of the
    coupling, which is easy to get backwards: finer grids need LOWER
    degree, because a smaller voxel is necessarily more isotropic.

    Worked values (f_max = fc + B/2):
      B787 npz (f_max 11.5 GHz), extent 0.15 g48 -> pitch 6.25 mm -> L=3
      B787 npz,                  extent 0.15 g96 -> pitch 3.13 mm -> L=2
      B787 npz,                  extent 0.06 g96 -> pitch 1.25 mm -> L=1
      PEC sphere (f_max 80.5 GHz), extent 1.5 g48 -> pitch 62.5 mm -> L=10 (capped)

    `cap` clamps to the representation's own ceiling (SHVoxelGridScene is
    built for degree <=10). `safety` >1 buys slack against this being a
    band-limit estimate rather than a hard cutoff; pruning a band is
    irreversible within a run only in the sense that growth must re-earn it.
    """
    if pitch <= 0:
        raise ValueError(f"pitch must be positive, got {pitch}")
    if f_max_hz <= 0:
        raise ValueError(f"f_max_hz must be positive, got {f_max_hz}")
    k_round_trip = 2.0 * (2.0 * math.pi * f_max_hz / c)
    a = (math.sqrt(3.0) / 2.0) * pitch
    return int(max(0, min(cap, math.ceil(safety * k_round_trip * a))))


def basis_degree_index(degree, device=None):
    """Returns a 1D int64 tensor of length num_sh_basis(degree) mapping
    each basis index to the SH degree l it belongs to (0,1,1,1,2,2,2,2,2,
    ...), matching the degree-major/m=-l..l ordering real_sh_basis and
    complex_sh_basis both use. Shared by SHVoxelGridScene (per-voxel order
    masking) and swe_operator (grouping (l,m) columns by l for the radial
    Bessel transform / Hankel propagator, which only depend on l)."""
    if not (0 <= degree <= MAX_SUPPORTED_DEGREE):
        raise ValueError(f"degree must be 0..{MAX_SUPPORTED_DEGREE}, got {degree}")
    flat = [l for l in range(degree + 1) for _ in range(2 * l + 1)]
    return torch.tensor(flat, dtype=torch.int64, device=device)


def _sh_normalization(l, am):
    """sqrt((2l+1)/(4pi) * (l-am)!/(l+am)!), computed via lgamma rather
    than math.factorial directly: factorial(l+am) alone overflows float64
    once l+am > ~170, which real_sh_basis/complex_sh_basis now legitimately
    reach (swe_operator.py needs L in the hundreds to thousands) even
    though the RATIO itself stays small and well-behaved. Caught by
    scripts/validate_swe_operator.py's off-center stage (l=150 already
    overflowed math.factorial(300)) -- not a hypothetical edge case."""
    log_k_sq = math.log((2 * l + 1) / (4 * math.pi)) + math.lgamma(l - am + 1) - math.lgamma(l + am + 1)
    return math.exp(0.5 * log_k_sq)


def _assoc_legendre(cos_theta, max_degree):
    """Returns {(l, m): P_l^m(cos_theta)} for all 0 <= m <= l <= max_degree.
    cos_theta is a 0-d tensor (single viewpoint); no gradient is needed
    through it (theta/phi come from the dataset, not a trained parameter),
    so the (1 - x^2) clamp below is purely for forward-pass safety at the
    poles (dtheta=0 does occur in the real AEDT sweep data) -- it does not
    need to be autograd-safe in reverse.
    """
    x = cos_theta
    somx2 = torch.clamp(1 - x * x, min=0).sqrt()  # sin(theta), clamped against fp noise at the poles

    P = {(0, 0): torch.ones_like(x)}
    fact = 1.0
    for m in range(1, max_degree + 1):
        P[(m, m)] = P[(m - 1, m - 1)] * (-fact) * somx2
        fact += 2.0
    for m in range(0, max_degree):
        P[(m + 1, m)] = x * (2 * m + 1) * P[(m, m)]
    for m in range(0, max_degree + 1):
        for l in range(m + 2, max_degree + 1):
            P[(l, m)] = (x * (2 * l - 1) * P[(l - 1, m)] - (l + m - 1) * P[(l - 2, m)]) / (l - m)
    return P


def real_sh_basis(theta, phi, degree):
    """Returns a 1D tensor of length (degree+1)**2 -- the real SH basis
    functions Y_lm evaluated at the single direction (theta, phi), ordered
    degree-major (see module docstring)."""
    if not (0 <= degree <= MAX_SUPPORTED_DEGREE):
        raise ValueError(f"degree must be 0..{MAX_SUPPORTED_DEGREE}, got {degree}")

    cos_theta = torch.cos(theta)
    P = _assoc_legendre(cos_theta, degree)

    basis = []
    for l in range(degree + 1):
        for m in range(-l, l + 1):
            am = abs(m)
            k = _sh_normalization(l, am)
            plm = P[(l, am)]
            if m == 0:
                val = k * plm
            elif m > 0:
                val = math.sqrt(2) * k * torch.cos(am * phi) * plm
            else:
                val = math.sqrt(2) * k * torch.sin(am * phi) * plm
            basis.append(val)
    return torch.stack(basis)


def complex_sh_basis(theta, phi, degree):
    """Complex spherical harmonics Y_lm, orthonormal on the unit sphere
    (integral |Y_lm|^2 dOmega = 1) with the standard Condon-Shortley phase
    -- same P_l^m table as real_sh_basis, so Y_l0 (always real) is
    numerically identical between the two functions. Unlike real_sh_basis,
    theta/phi may be any matching shape (not just a single scalar
    viewpoint): returns a complex tensor of shape
    (num_sh_basis(degree), *theta.shape), degree-major/m=-l..l ordering
    (same convention as real_sh_basis). Used by rift/swe_operator.py,
    where genuinely complex Y_lm (not the real basis used for
    SHVoxelGridScene's angular reflectivity) match the spherical-wave
    literature's convention.
    """
    if not (0 <= degree <= MAX_SUPPORTED_DEGREE):
        raise ValueError(f"degree must be 0..{MAX_SUPPORTED_DEGREE}, got {degree}")

    cos_theta = torch.cos(theta)
    P = _assoc_legendre(cos_theta, degree)

    basis = []
    for l in range(degree + 1):
        for m in range(-l, l + 1):
            am = abs(m)
            k = _sh_normalization(l, am)
            val = k * P[(l, am)] * torch.exp(1j * m * phi)
            basis.append(val)
    return torch.stack(basis)
