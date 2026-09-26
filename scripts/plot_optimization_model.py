#!/usr/bin/env python
"""Render the RIFT optimization model as equation panels for the slide deck.

Three panels, matching what train.py actually optimizes:
  scene   -- rift/sparse_scene.py::AdaptiveSHVoxelGrid.active_scatterers
  signal  -- rift/forward_operator.py::forward_operator_lessparallel (+ range_operator)
  objective -- train.py::viewpoint_loss + regularization_loss, rift/calibration.py

    module load anaconda3 && conda activate RIFT
    python scripts/plot_optimization_model.py --outdir figures/b787_recon
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
ACCENT = "#2a78d6"

PANELS = {
    "paper_method_scene": [
        (0.28, r"$\mathbf{Scene\ model}$  —  a field of direction-dependent point scatterers",
         15, ACCENT),
        (1.02, r"$\mathcal{S}=\{(p_n,c^n)\}_{n=1}^{N}, \qquad "
               r"\rho_n(\hat d)=\sum_{\ell=0}^{L_n}\sum_{m=-\ell}^{\ell}"
               r"c^{\,n}_{\ell m}\,Y_{\ell m}(\hat d)$", 23, INK),
        (1.85, r"$p_n\in\mathbb{R}^3$: scatterer location;   "
               r"$Y_{\ell m}$: real spherical-harmonic basis;   "
               r"$c^{\,n}_{\ell m}\in\mathbb{C}$: learned amplitude and phase", 13, MUTED),
        (2.38, r"$\mathbf{The\ SH\ expansion\ models\ each\ point's\ complex\ reflectivity}$",
         16, INK),
        (2.78, r"$\mathbf{as\ a\ function\ of\ viewing\ direction.}$", 16, INK),
        (3.32, r"one fitted field", 14, ACCENT),
        (3.72, r"held-out direction $\hat d$  $\longrightarrow$  coherent radar response", 14, MUTED),
        (4.12, r"spatial support $\{p_n:\sum_{\ell m}|c^{\,n}_{\ell m}|^2>\tau\}$  "
               r"$\longrightarrow$  metric 3-D reconstruction", 14, MUTED),
        (4.72, r"$\mathbf{A\ voxel\ is\ not\ the\ scene\ model.}$", 15, INK),
        (5.12, r"The locations $p_n$ may be sampled on a grid or by another point layout; "
               r"the mathematics is unchanged.", 13, MUTED),
    ],
    "paper_method_signal": [
        (0.28, r"$\mathbf{Signal\ model}$  —  radar measurements are Fourier samples of the field",
         15, ACCENT),
        (1.02, r"$S_v(f)[r,t]=\sum_n\frac{\rho_n(\hat d_v)}"
               r"{(R^{v,r,t}_{\Sigma,n})^2}\,"
               r"\exp\!\left(s\,i\,\frac{2\pi f}{c}\,R^{v,r,t}_{\Sigma,n}\right)$",
         23, INK),
        (1.86, r"$R^{v,r,t}_{\Sigma,n}=\|p_n-t^v_t\|+\|p_n-r^v_r\|$: exact bistatic path "
               r"Tx $\rightarrow$ point $\rightarrow$ Rx", 13, MUTED),
        (2.52, r"Let $\tau_n=R_{\Sigma,n}/c$ and "
               r"$a_n=\rho_n(\hat d_v)/R_{\Sigma,n}^{2}$:", 14, MUTED),
        (3.08, r"$S_v(f)[r,t]=\sum_n a_n\,e^{\,s i 2\pi f\tau_n}$", 23, INK),
        (3.72, r"$\mathbf{For\ one\ view\ and\ channel,\ each\ point\ contributes\ one\ complex\ "
               r"sinusoid\ across\ frequency.}$", 15, INK),
        (4.26, r"Their coherent sum is Fourier synthesis of the direction-conditioned "
               r"reflectivity over bistatic delay.", 14, MUTED),
        (4.78, r"A new viewpoint changes both $\rho_n(\hat d_v)$ and the exact path delays; "
               r"the same equation therefore synthesizes an unseen radar view.", 13, MUTED),
    ],
    "paper_method_implementation": [
        (0.28, r"$\mathbf{Implementation}$  —  fast exact evaluation; sampling stays separate",
         15, ACCENT),
        (0.92, r"nonuniform delays $\{\tau_n\}$", 15, MUTED),
        (1.44, r"$\longrightarrow$  Gaussian spreading  $\longrightarrow$  oversampled 1-D FFT  "
               r"$\longrightarrow$  $S(f_i)$ on uniform swept frequencies", 17, INK),
        (2.18, r"The range-factorized type-1 NUFFT accelerates the frequency axis:", 14, MUTED),
        (2.68, r"$O(NN_f)$ direct synthesis  $\longrightarrow$  "
               r"$O(NJ+M\log M)$, with a small gridding support $J$", 20, INK),
        (3.34, r"$\mathbf{Exact\ near-field\ geometry\ is\ retained:}$ no far-field "
               r"approximation and no spatial interpolation in the training path.", 14, INK),
        (4.02, r"$\mathbf{Sampling\ choice:}$ select the locations $p_n$. Our experiments use "
               r"voxel centres on a uniform lattice;", 13, MUTED),
        (4.40, r"the forward operator itself accepts arbitrary point locations.", 13, MUTED),
        (4.88, r"$\min_{c,g}\sum_v\|g\,F_v(c)-S_v\|_2^2"
               r"+\mathcal{R}(c)$", 20, INK),
        (5.46, r"Training fits the SH coefficients and one global complex gain. The grid is a "
               r"discretization—not a learned surface or SDF prior.", 13, MUTED),
    ],
    "opt_model_scene": [
        (0.30, r"$\mathbf{Scene\ model}$  —  adaptive spherical-harmonic voxel grid "
               r"(`grid_sh`)", 15, ACCENT),
        (0.95, r"$w_n(\hat{d}) \;=\; \sum_{\ell=0}^{L_n}\ \ \sum_{m=-\ell}^{\ell} "
               r"c^{\,n}_{\ell m}\; Y_{\ell m}(\hat{d}), \qquad c^{\,n}_{\ell m}\in\mathbb{C}$", 25, INK),
        (1.75, r"voxel centres $p_n$ frozen on a $G^3$ lattice over $[-e,e]^3$   "
               r"($G=48$, $e=0.15$ m)", 14, MUTED),
        (2.20, r"$\hat{d}$ = monostatic look direction of the viewpoint  →  "
               r"view-dependent reflectivity", 14, MUTED),
        (2.65, r"free parameters: $\mathrm{Re}\,c^{\,n}_{\ell m},\ \mathrm{Im}\,c^{\,n}_{\ell m}$   "
               r"(+ per-voxel active mask and order $L_n$)", 14, MUTED),
        (3.30, r"$\mathbf{Angular\ capacity\ is\ capped\ by\ physics:}\;\; "
               r"L \;\leq\; 2\,k_{max}\,a$", 17, INK),
        (3.85, r"half-diagonal $a$ of a voxel; the round-trip $2k$ is what the SH argument sees. "
               r"g48/e0.15 $\Rightarrow L=3$.", 13, MUTED),
        (4.30, r"No surface/SDF prior, and no trilinear interpolation in the training path.", 13, MUTED),
    ],
    "opt_model_signal": [
        (0.30, r"$\mathbf{Signal\ model}$  —  differentiable physics-based radar forward operator", 15, ACCENT),
        (1.00, r"$S(f)[r,t] \;=\; \sum_n\, w_n(\hat{d})\;\cdot\;"
               r"\frac{1}{R_{\Sigma}^{\,2}}\;\cdot\;"
               r"\exp\!\left(\,s\; i\, k\, R_{\Sigma}\right)$", 25, INK),
        (1.90, r"$R_{\Sigma} \;=\; \|p_n - t_t\| + \|p_n - r_r\|$   (bistatic path, "
               r"element $\rightarrow$ scatterer $\rightarrow$ element)", 14, MUTED),
        (2.35, r"$k = 2\pi f/c$   over 600 swept frequencies;   16 Tx $\times$ 16 Rx MIMO,   "
               r"output $[n_f, N_{rx}, N_{tx}]$", 14, MUTED),
        (2.80, r"$s = $ phase sign $= -1$ for every FMCW npz dataset "
               r"(opposite the CSV/AEDT convention)", 14, MUTED),
        (3.40, r"$\mathbf{Complex\ weight\ }w_n\mathbf{\ enters\ LINEARLY}$ — never "
               r"$|w|$ or $\arg w$", 16, INK),
        (3.90, r"an empty scene would otherwise be a zero-gradient fixed point.", 13, MUTED),
        (4.35, r"Evaluated by the range-factorized 1-D NUFFT operator (exact in the near field, "
               r"fp64 phase path).", 13, MUTED),
    ],
    "opt_model_objective": [
        (0.28, r"$\mathbf{Global\ gain}$  —  one complex scalar per run, learned jointly "
               r"with the scene", 15, ACCENT),
        (0.92, r"$\hat{S}_v \;=\; g \cdot F_v(w), \qquad "
               r"g \;=\; e^{\,\alpha}\, e^{\,i\psi} \;\in\; \mathbb{C}$", 23, INK),
        (1.62, r"Deliberately GLOBAL, never per-viewpoint: a per-view gain would absorb the "
               r"relative phase", 13, MUTED),
        (2.00, r"between viewpoints and destroy the cross-view coherence that gives RIFT its "
               r"resolution.", 13, MUTED),
        (2.38, r"Log-magnitude parametrized, warm-started from $\langle \hat S, S\rangle / "
               r"\|\hat S\|^2$ on the first view.", 13, MUTED),
        (2.95, r"$\mathbf{Objective}$", 15, ACCENT),
        (3.52, r"$\min_{c,\,\alpha,\,\psi}\;\; \sum_{v}\left\langle \left| g\,F_v(w) - S_v "
               r"\right|^2 \right\rangle \;+\; \lambda_1 |g| \left\langle \|c^{\,n}\|_2 "
               r"\right\rangle_n \;+\; \lambda_2 |g|^2 \left\langle \sum_{\ell m} "
               r"\ell(\ell+1)\,|c^{\,n}_{\ell m}|^2 \right\rangle_n$", 16, INK),
        (4.30, r"group-L1 sparsity ($\lambda_1 = 3\!\times\!10^{-7}$) + angular Laplacian; the "
               r"$|g|$ factors make both terms gauge-invariant.", 13, MUTED),
        (4.80, r"$\mathbf{Reported\ metric:}\;\; \mathrm{rel\text{-}MSE} = "
               r"\sum_v \|\hat S_v - S_v\|^2 \,/\, \sum_v \|S_v\|^2$   on held-out viewpoints.", 15, INK),
    ],
}


def render(name, lines, outdir):
    height = max(y for y, _, _, _ in lines) + 0.55
    fig = plt.figure(figsize=(12.2, height))
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 12.2)
    ax.set_ylim(height, 0)       # y grows downward, so entries read top-to-bottom
    ax.axis("off")
    for y, text, size, color in lines:
        ax.text(0.35, y, text, fontsize=size, color=color, va="center", ha="left")
    out = os.path.join(outdir, name + ".png")
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="figures/b787_recon")
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    for name, lines in PANELS.items():
        render(name, lines, args.outdir)


if __name__ == "__main__":
    main()
