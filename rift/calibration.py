"""Learnable global calibration between the forward operator and measured data.

The forward operator's absolute scale is not physically calibrated against
AEDT/HFSS S-parameter port normalization (its 1/(4pi)^2 constant and
geometric-gain heuristic leave a large constant magnitude/phase offset --
measured ~550x on the AEDT sphere data). Rather than derive that constant,
a single GLOBAL complex gain g = exp(log_mag) * exp(i*phase) is learned
jointly with the scene: S_pred = g * forward_operator(...).

Deliberately global, NOT per-viewpoint: a per-viewpoint complex gain would
absorb the relative phase between viewpoints and destroy exactly the
cross-viewpoint coherence (the multi-view synthetic aperture) that gives
RIFT its resolution. Keep it one scalar per training run.

log-magnitude parametrization so Adam-scale steps move the gain
multiplicatively (reaching a ~550x calibration factor takes ~ln(550)/lr
steps instead of ~550/lr).
"""
import torch
import torch.nn as nn

from rift.distributed import rank0_print


class GlobalComplexGain(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_mag = nn.Parameter(torch.zeros(()))
        self.phase = nn.Parameter(torch.zeros(()))
        # Flipped after the one-time magnitude warm start; persisted in
        # checkpoints so resumed runs don't re-initialize.
        self.register_buffer("initialized", torch.tensor(False))

    def forward(self, s):
        return torch.polar(torch.exp(self.log_mag), self.phase) * s

    def gain_value(self):
        """Current complex gain as a python complex (for logging)."""
        with torch.no_grad():
            return complex(torch.polar(torch.exp(self.log_mag), self.phase).item())

    @torch.no_grad()
    def maybe_init_scale(self, s_pred_raw, s_meas):
        """One-time warm start of the gain from the first rendered viewpoint.

        If the prediction is meaningfully correlated with the measurement
        (e.g. after backprojection init), uses the loss-optimal complex
        projection g = <S_pred, S_meas> / ||S_pred||^2 -- magnitude AND
        phase. For an uncorrelated (random-init) scene the projection is
        ~0 and would kill the signal, so it falls back to matching power
        only: |g| = ||S_meas|| / ||S_pred||, phase 0.
        """
        if bool(self.initialized):
            return
        meas_norm = torch.linalg.vector_norm(s_meas)
        pred_norm = torch.linalg.vector_norm(s_pred_raw)
        if pred_norm < 1e-20:
            # Zero-initialized scene: no scale information yet. Stay pending
            # so a later viewpoint (after the first optimizer step) warms up.
            return
        if s_pred_raw.shape != s_meas.shape:
            raise ValueError(f"maybe_init_scale needs identically-laid-out tensors, got "
                             f"{tuple(s_pred_raw.shape)} vs {tuple(s_meas.shape)}")
        proj = (s_pred_raw.conj() * s_meas).sum() / (pred_norm ** 2).clamp_min(1e-30)
        norm_ratio = (meas_norm / pred_norm.clamp_min(1e-30)).clamp_min(1e-30)
        # |proj| / norm_ratio is the correlation coefficient between
        # prediction and measurement.
        if proj.abs() > 0.05 * norm_ratio:
            self.log_mag.fill_(torch.log(proj.abs().clamp_min(1e-30)).item())
            self.phase.fill_(torch.angle(proj).item())
            how = "projection <S_pred,S_meas>/||S_pred||^2"
        else:
            self.log_mag.fill_(torch.log(norm_ratio).item())
            how = "power ratio ||S_meas||/||S_pred|| (prediction uncorrelated with data)"
        self.initialized.fill_(True)
        g = torch.polar(torch.exp(self.log_mag), self.phase)
        rank0_print(f"GlobalComplexGain: warm-started g = {complex(g.item()):.4e} via {how}")
