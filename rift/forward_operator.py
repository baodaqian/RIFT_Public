"""Differentiable physics-based radar forward model.

Given a set of complex scatterer weights on a 3D grid and a virtual MIMO
array pose, renders the frequency-domain S-parameter response that array
would observe -- the "rendering" step of the analysis-by-synthesis loop
(scene MLP -> forward operator -> compare to measured radar data).
"""
import torch


def get_kvector(freqs, cc):
    """freqs must be in Hz."""
    return 2 * torch.pi * freqs / cc


def forward_operator_lessparallel(
    freqs, kvector, arr_pos_rx, arr_pos_tx,
    scatterer_pos, scatterer_weights,
    artificial_gain: float = 1.0,
    p_spectrum: torch.Tensor = None,
    range_model: str = "sum2",
    omega_scaling: str = "unity",
    center_freq_hz: float = None,
    eps: float = 1e-9,
    phase_sign: float = 1.0,
):
    """
    Memory-light version: loops over frequency channels.
    Returns S(omega) with shape [nf, num_rx, num_tx] (complex).
    Includes an eps stabilization term in the geometric gain to avoid
    division-by-zero when a scatterer coincides with an array element.

    phase_sign: sign of the propagation phase exp(phase_sign * i * k * R).
    THE SIGN IS A PER-SIMULATOR CONVENTION -- verify it for every new data
    source with scripts/check_phase_sign.py. +1 (default) is confirmed for
    this project's AEDT/HFSS export pipeline (sphere + B787 data) and is
    also the convention expected for future Ansys AVXcelerate data.

    BEWARE THE FREQUENCY-GRID ALIAS when validating a sign: on uniformly
    sampled frequencies (spacing df), (phase_sign, range R) and
    (-phase_sign, m*c/(2*df) - R) produce IDENTICAL single-viewpoint data up
    to a constant phase, and BOTH this project's targets happen to sit at
    the alias-symmetric range (sphere: 7m vs 8m of the 15m window; B787:
    ~50m of the 50m window) -- so naive range fits cannot distinguish the
    signs here. The reliable discriminator is range-profile causality (no
    scattering can arrive EARLIER than the first surface; see
    scripts/check_phase_sign.py), which is how +1 was established
    2026-07-03. The sign matters critically for multi-viewpoint training on
    NON-spherical targets: the wrong sign mirrors the scene differently per
    viewpoint, so no single scene can fit all viewpoints.
    """
    device    = scatterer_pos.device
    out_dtype = torch.cfloat

    nf     = freqs.shape[0]
    N      = scatterer_pos.shape[0]
    num_rx = arr_pos_rx.shape[0]
    num_tx = arr_pos_tx.shape[0]

    d_tx = scatterer_pos[:, None, :] - arr_pos_tx[None, :, :]   # [N, Tx, 3]
    d_rx = scatterer_pos[:, None, :] - arr_pos_rx[None, :, :]   # [N, Rx, 3]
    R_tx = torch.linalg.norm(d_tx, dim=-1).clamp_min(eps)       # [N, Tx]
    R_rx = torch.linalg.norm(d_rx, dim=-1).clamp_min(eps)       # [N, Rx]

    if range_model == "product":
        G_geom = 1.0 / (R_tx[:, :, None] * R_rx[:, None, :] + eps)          # [N, Tx, Rx]
    elif range_model == "sum2":
        G_geom = 1.0 / ((R_tx[:, :, None] + R_rx[:, None, :]).pow(2) + eps) # [N, Tx, Rx]
    elif range_model == "none":
        G_geom = torch.ones((N, num_tx, num_rx), dtype=freqs.dtype, device=device)
    else:
        raise ValueError("range_model must be one of {'product','sum2','none'}")

    # The complex weight enters LINEARLY (w * exp(i*phase_geom)), never
    # decomposed into abs/angle: torch.abs/torch.angle are non-differentiable
    # at w=0, which made an all-zero scene a fixed point of training (zero
    # gradient) and gave near-zero voxels unusable gradients. Mathematically
    # identical to the old w_amp*exp(i*(phase_geom + w_phase)) everywhere
    # else.
    w_cplx = scatterer_weights.to(torch.cfloat).view(N, 1, 1)               # [N,1,1]

    freqs   = freqs.to(device)
    kvector = kvector.to(device)
    omega   = 2.0 * torch.pi * freqs                                       # [nf]

    if p_spectrum is None:
        p_mag_all = None
        p_phi_all = None
    else:
        p = torch.as_tensor(p_spectrum, device=device)
        if torch.is_complex(p):
            p_mag_all = torch.abs(p).to(freqs.dtype)                        # [nf]
            p_phi_all = torch.angle(p).to(freqs.dtype)                      # [nf]
        else:
            p_mag_all = p.abs().to(freqs.dtype)                             # [nf]
            p_phi_all = torch.zeros_like(p_mag_all)                         # [nf]

    if omega_scaling == "center":
        fc = freqs.mean() if center_freq_hz is None else torch.tensor(center_freq_hz, device=device, dtype=freqs.dtype)
        omega_center_sq = (2.0 * torch.pi * fc) ** 2
    g_const = 1.0 / ((4.0 * torch.pi) ** 2)

    out = torch.empty((nf, num_rx, num_tx), dtype=out_dtype, device=device)

    for i in range(nf):
        omega_i = omega[i]
        k_i     = kvector[i]
        if omega_scaling == "frequency":
            omega_factor_i = omega_i * omega_i
        elif omega_scaling == "center":
            omega_factor_i = omega_center_sq
        elif omega_scaling == "unity":
            omega_factor_i = torch.as_tensor(1.0, device=device, dtype=freqs.dtype)
        else:
            raise ValueError("omega_scaling must be one of {'frequency','center','unity'}")

        p_mag_i = (p_mag_all[i] if p_mag_all is not None else torch.as_tensor(1.0, device=device, dtype=freqs.dtype))
        p_phi_i = (p_phi_all[i] if p_phi_all is not None else torch.as_tensor(0.0, device=device, dtype=freqs.dtype))

        A_i = (
            artificial_gain
            * g_const
            * omega_factor_i
            * p_mag_i
            * G_geom
        ).to(freqs.dtype)

        R_sum = R_tx[:, :, None] + R_rx[:, None, :]    # [N, Tx, Rx]
        phase_i = (
            phase_sign * k_i * R_sum
            + p_phi_i
        )

        field_i = (w_cplx * A_i) * torch.exp(1j * phase_i)  # [N, Tx, Rx], complex
        S_tx_rx = field_i.sum(dim=0)                        # [Tx, Rx]
        out[i]  = S_tx_rx.transpose(0, 1).to(out_dtype)     # [Rx, Tx]

        del R_sum, phase_i, field_i, S_tx_rx

    return out  # [nf, Rx, Tx]


def adjoint_operator_lessparallel(
    freqs, kvector, arr_pos_rx, arr_pos_tx,
    scatterer_pos, S_resid,
    range_model: str = "sum2",
    eps: float = 1e-9,
    phase_sign: float = 1.0,
):
    """Adjoint (conjugate transpose) of forward_operator_lessparallel with
    respect to the scatterer weights, for the default unity omega-scaling /
    no-pulse-spectrum configuration train.py uses:

        w[N] = sum_{f,rx,tx} conj(kernel[f,rx,tx,n]) * S_resid[f,rx,tx]

    Applied to measured data this is the matched-filter / backprojection
    image -- the classical coherent SAR image on the voxel grid -- used as a
    physics-informed initialization. Must use the same phase_sign as the
    forward operator.
    """
    device = scatterer_pos.device
    N = scatterer_pos.shape[0]
    nf = freqs.shape[0]

    d_tx = scatterer_pos[:, None, :] - arr_pos_tx[None, :, :]
    d_rx = scatterer_pos[:, None, :] - arr_pos_rx[None, :, :]
    R_tx = torch.linalg.norm(d_tx, dim=-1).clamp_min(eps)       # [N, Tx]
    R_rx = torch.linalg.norm(d_rx, dim=-1).clamp_min(eps)       # [N, Rx]
    R_sum = R_tx[:, :, None] + R_rx[:, None, :]                 # [N, Tx, Rx]

    if range_model == "product":
        G_geom = 1.0 / (R_tx[:, :, None] * R_rx[:, None, :] + eps)
    elif range_model == "sum2":
        G_geom = 1.0 / (R_sum.pow(2) + eps)
    elif range_model == "none":
        G_geom = torch.ones_like(R_sum)
    else:
        raise ValueError("range_model must be one of {'product','sum2','none'}")

    g_const = 1.0 / ((4.0 * torch.pi) ** 2)
    kvector = kvector.to(device)

    w = torch.zeros(N, dtype=torch.cfloat, device=device)
    for i in range(nf):
        # conj of the forward kernel: G * exp(-i * phase_sign * k * R_sum)
        ker_i = (g_const * G_geom) * torch.exp(-1j * phase_sign * kvector[i] * R_sum)  # [N, Tx, Rx]
        w += torch.einsum("ntr,rt->n", ker_i, S_resid[i].to(torch.cfloat))
    return w


def get_array_pos(theta, phi, array_dist, spacing, num_rx, num_tx, device):
    """Absolute position of Tx/Rx array elements in the global (object) frame."""
    theta = theta.squeeze(0)[0]
    phi = phi.squeeze(0)[0]

    r_0 = array_dist * torch.tensor([torch.sin(theta) * torch.cos(phi),
                                     torch.sin(theta) * torch.sin(phi),
                                     torch.cos(theta)], device=device)

    rx_positions = torch.zeros(num_rx, 3, device=device)
    tx_positions = torch.zeros(num_tx, 3, device=device)

    for n in range(num_rx):
        offset = spacing * (n + 0.5) * torch.tensor([-torch.sin(phi), torch.cos(phi), 0], device=device)
        rx_positions[n] = r_0 + offset

    for n in range(num_tx):
        offset = spacing * (n + 0.5) * torch.tensor([-torch.cos(theta) * torch.cos(phi),
                                                    -torch.cos(theta) * torch.sin(phi),
                                                     torch.sin(theta)], device=device)
        tx_positions[n] = r_0 + offset

    return rx_positions, tx_positions
