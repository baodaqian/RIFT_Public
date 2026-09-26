#!/usr/bin/env python
"""Validation rel-MSE vs epoch for the B787 novel-view-synthesis runs.

Scrapes the per-epoch curves straight out of the SLURM logs (via
scrape_training_logs) and draws the capacity-control story: the degree-6
baseline against the SH-capped / pruned arms.

    module load anaconda3 && conda activate RIFT
    python scripts/plot_val_error_curves.py --out figures/b787_recon/val_error_curves.png
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.scrape_training_logs import DEFAULT_LOG_GLOBS, scrape  # noqa: E402

# Categorical slots 1,2,3,4,7 of the validated palette (light mode).
BEST_RUN = "b787_r5_prune_target20k"

SERIES = [
    ("b787_sphere2k_gridsh6_g48_n1800", "SH degree 6 (baseline, uncapped)", "#4a3aa7"),
    ("b787_r5_deg0",                    "SH degree 0 (isotropic)",          "#eda100"),
    ("b787_r4_capdeg3",                 "SH degree 3 (physics cap)",        "#1baf7a"),
    ("b787_r5_deg2",                    "SH degree 2",                      "#eb6834"),
    ("b787_r5_prune_target20k",         "SH degree 3 + prune to 20k voxels", "#2a78d6"),
]

# Round 6 (2026-08-06--08): occlusion + SpINRv2 staged magnitude supervision,
# all arms on the R4 deg-3 config. The set draws TRAIN as well as VAL because
# the two SpINRv2 arms fit train harder than the baseline and generalize worse;
# that conclusion is only visible with both panels side by side.
ROUND6 = [
    ("b787_r4_capdeg3",   "no occlusion (baseline, deg 3)", "#52514e"),
    ("b787_r6_occ_z0p1",  "occlusion · ζ frozen 0.1",       "#eda100"),
    ("b787_r6_occ_z1p0",  "occlusion · ζ frozen 1.0",       "#d03b3b"),
    ("b787_r6_occ_learn", "occlusion · ζ learned (energy)", "#1baf7a"),
    ("b787_r6_occ_dc",    "occlusion · ζ learned (DC — SH-SAS keying)", "#2a78d6"),
    ("b787_r6_magmix",    "SpINRv2 magnitude λ=2, no warm-up", "#eb6834"),
    ("b787_r6_magwarm",   "SpINRv2 magnitude λ=2 + 15-epoch warm-up", "#a3459b"),
]

CURRENT_BEST = [
    ("b787_r5_prune_target20k", "Round 5 best · degree 3 + target 20k", "#2a78d6"),
    ("b787_r6_occ_dc", "Round 6 best · learned DC opacity", "#1baf7a"),
]


def style(ax):
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, axis="y", color="#e6e5e2", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c4")
    ax.tick_params(colors="#52514e", labelsize=10.5)


def plot_train_val(runs, out, series, title):
    """Two panels — train rel-MSE (log) and val rel-MSE — for selected runs."""
    fig, (axt, axv) = plt.subplots(1, 2, figsize=(13.2, 6.0))
    fig.patch.set_facecolor("#fcfcfb")
    legend_entries = []

    for key, label, color in series:
        eps = runs.get(key, [])
        if not eps:
            continue
        tr = sorted((d["epoch"], d["train"]) for d in eps if "train" in d)
        va = sorted((d["epoch"], d["val"]) for d in eps if "val" in d)
        lw = 2.6 if key == "b787_r4_capdeg3" else 2.0
        ls = (0, (4, 3)) if key == "b787_r4_capdeg3" else "solid"
        axt.plot([a for a, _ in tr], [b for _, b in tr], color=color,
                 linewidth=lw, linestyle=ls, solid_capstyle="round", zorder=3)
        axv.plot([a for a, _ in va], [b for _, b in va], color=color,
                 linewidth=lw, linestyle=ls, solid_capstyle="round", zorder=3)
        best = min(b for _, b in va)
        ep_best = [a for a, b in va if b == best][0]
        axv.plot([ep_best], [best], marker="o", ms=6.0, color=color,
                 markeredgecolor="#fcfcfb", markeredgewidth=1.5, zorder=4)
        legend_entries.append((f"{label} — best val {best:.1f}%", color))

    axt.set_yscale("log")
    axt.set_ylabel("training rel-MSE  (log scale)", fontsize=12, color="#52514e")
    axt.set_title("Training error", fontsize=12.5, color="#0b0b0b", loc="left", pad=8)
    axv.set_ylabel("validation rel-MSE  (held-out viewpoints)", fontsize=12, color="#52514e")
    axv.set_title("Validation error — held-out viewpoints", fontsize=12.5,
                  color="#0b0b0b", loc="left", pad=8)
    axv.set_ylim(20, 108)
    for ax in (axt, axv):
        ax.set_xlabel("epoch", fontsize=12, color="#52514e")
        ax.set_xlim(0, 152)
        style(ax)
    for ax in (axt, axv):
        ax.axhline(100, color="#d03b3b", linewidth=1.0, linestyle=(0, (5, 4)), zorder=2)
    axv.annotate("predict-zero floor (100%)", xy=(150, 100), xytext=(0, 5),
                 textcoords="offset points", fontsize=10, color="#d03b3b", ha="right")

    fig.legend([plt.Line2D([], [], color=c, linewidth=2.2) for _, c in legend_entries],
               [l for l, _ in legend_entries], loc="lower center", frameon=False,
               fontsize=10.5, labelcolor="#0b0b0b", ncol=2)
    fig.suptitle(title,
                 fontsize=13.5, color="#0b0b0b", x=0.012, ha="left")
    fig.tight_layout(rect=[0, 0.155, 1, 0.955])
    fig.savefig(out, dpi=170, facecolor=fig.get_facecolor())
    print("wrote", out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", nargs="+", default=list(DEFAULT_LOG_GLOBS),
                   help="glob(s) of SLURM logs to scrape (run from the repo root); "
                        "default covers the root and archive/slurm_reports/by_run/")
    p.add_argument("--set", choices=("capacity", "round6", "current_best"), default="capacity")
    p.add_argument("--out", default="figures/b787_recon/val_error_curves.png")
    args = p.parse_args()

    runs = {k: v["epochs"] for k, v in scrape(args.logs).items()}
    if args.set == "round6":
        plot_train_val(
            runs, args.out, ROUND6,
            "Round 6 — occlusion and SpINRv2 magnitude supervision, B787 "
            "(all seven runs complete at 150 epochs)",
        )
        return
    if args.set == "current_best":
        plot_train_val(
            runs, args.out, CURRENT_BEST,
            "B787 current best — Round 5 target-20k vs Round 6 learned-DC",
        )
        return
    legend_entries = []

    fig, ax = plt.subplots(figsize=(11, 6.2))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    for key, label, color in SERIES:
        pts = [(d["epoch"], d["val"]) for d in runs[key] if "val" in d]
        if not pts:
            continue
        pts.sort()
        ep = [a for a, _ in pts]
        val = [b for _, b in pts]
        ax.plot(ep, val, color=color, linewidth=2.0, solid_capstyle="round", zorder=3)
        best = min(val)
        legend_entries.append((f"{label} — best {best:.1f}%", color))
        # mark the best epoch; the value itself rides in the legend, so the
        # crowded low-error bundle carries no colliding end-labels
        ax.plot([ep[val.index(best)]], [best], marker="o", ms=6.5, color=color,
                markeredgecolor="#fcfcfb", markeredgewidth=1.6, zorder=4)
        if key == BEST_RUN:
            ax.annotate(f"best so far: {best:.1f}%",
                        xy=(ep[val.index(best)], best), xytext=(-14, -30), ha="right",
                        textcoords="offset points", fontsize=11.5, color="#0b0b0b",
                        arrowprops=dict(arrowstyle="-", color="#52514e", linewidth=1.0))

    ax.set_xlabel("epoch", fontsize=12, color="#52514e")
    ax.set_ylabel("validation rel-MSE  (held-out viewpoints)", fontsize=12, color="#52514e")
    ax.set_ylim(20, 105)
    ax.set_xlim(0, 158)
    ax.grid(True, axis="y", color="#e6e5e2", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c4")
    ax.tick_params(colors="#52514e", labelsize=10.5)

    # predict-zero floor: the failure mode every collapsed run sits at
    ax.axhline(100, color="#d03b3b", linewidth=1.2, linestyle=(0, (5, 4)), zorder=2)
    ax.annotate("predict-zero floor (100%)", xy=(3, 100), xytext=(0, 5),
                textcoords="offset points", fontsize=10, color="#d03b3b")

    ax.legend([plt.Line2D([], [], color=c, linewidth=2.0) for _, c in legend_entries],
              [l for l, _ in legend_entries], loc="upper right", frameon=False,
              fontsize=11, labelcolor="#0b0b0b")

    ax.set_title("B787 novel-view synthesis: capping angular capacity is what buys generalization",
                 fontsize=13.5, color="#0b0b0b", pad=14, loc="left")
    fig.tight_layout()
    fig.savefig(args.out, dpi=170, facecolor=fig.get_facecolor())
    print("wrote", args.out)


if __name__ == "__main__":
    main()
