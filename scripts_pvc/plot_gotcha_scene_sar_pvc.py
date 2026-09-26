#!/usr/bin/env python3
"""Appendix figure (fig:gotcha-scene-sar): the measured GOTCHA scene with the three vehicles circled.

The image is Casteel et al. (2007), Fig. 1: a 2D SAR image of the spotlighted scene centre from one pass with 360
degrees of aperture, in the native GOTCHA frame (x, y in m, +-50 m). Reuse with citation confirmed by the user
(2026-09-25). Only its plot area is used: each pixel's jet colour is mapped back to its jet level (nearest colour) and
redrawn on inferno, the paper's single colour scale (brighter = larger); the axes are redrawn in the paper's Times.
The circles are centred on each vehicle's region of interest, the translation of its region definition in
rift_pvc/regions/ (the App. 8 origins); the published figure places the scene's features to about +-1 m in that
frame (docs/RIFT_GOTCHA_Tune.md A73).

    python scripts_pvc/plot_gotcha_scene_sar_pvc.py --out-dir DIR --formats pdf,svg,png
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts_pvc.paper_figure_font import retarget_svg_family, use_paper_font  # noqa: E402

ACCENT, MUTED = "#dd513a", "#52514e"
SOURCE = Path("/scratch/group/p.cis261724.000/RIFT_pvc_runs/gotcha_new_targets_20260924/casteel2007_fig1_scene_sar.jpg")
BOX = (80, 813, 31, 764)          # the published axes box in pixels: columns x = -50..50 m, rows y = 50..-50 m
REGIONS = [("camry_box_v2", "rift_pvc/regions/camry_box_v2.json", "Toyota Camry"),
           ("sentra_box_v1", "rift_pvc/regions/gotcha_new_targets.json", "Nissan Sentra"),
           ("santafe_box_v1", "rift_pvc/regions/gotcha_new_targets.json", "Hyundai\nSanta Fe")]
# label anchor (native m) and alignment per vehicle, placed in empty parts of the scene
LABELS = {"camry_box_v2": ((15.5, -12.0), "right"), "santafe_box_v1": ((33.0, -11.0), "left"),
          "sentra_box_v1": ((28.0, -35.5), "left")}
RADIUS = 4.0


def jet_to_level(rgb):
    """Nearest jet colour -> level in [0, 1] (the published image's own scale, order preserved)."""
    lut = plt.get_cmap("jet")(np.linspace(0, 1, 256))[:, :3]
    flat = rgb.reshape(-1, 3)
    level = np.empty(len(flat))
    for start in range(0, len(flat), 65536):
        block = flat[start:start + 65536]
        level[start:start + 65536] = np.argmin(((block[:, None, :] - lut[None]) ** 2).sum(-1), 1) / 255.0
    return level.reshape(rgb.shape[:2])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--formats", default="pdf,svg,png")
    p.add_argument("--width", type=float, default=3.3, help="figure width (in)")
    args = p.parse_args()
    use_paper_font()
    plt.rcParams.update({"font.size": 7, "axes.edgecolor": MUTED, "axes.linewidth": 0.5, "xtick.color": MUTED,
                         "ytick.color": MUTED, "xtick.labelsize": 6, "ytick.labelsize": 6})
    image = mpimg.imread(SOURCE)[..., :3].astype(float) / 255.0
    c0, c1, r0, r1 = BOX
    level = jet_to_level(image[r0 + 1:r1, c0 + 1:c1])
    vehicles = []
    for key, path, name in REGIONS:
        region = json.loads(Path(path).read_text())["regions"][key]
        vehicles.append((key, name, np.array(region["translation_m"][:2])))

    W = args.width
    left, right, bottom, top = 0.38, 0.08, 0.33, 0.06                # margins (in)
    side = W - left - right
    H = side + bottom + top
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([left / W, bottom / H, side / W, side / H])
    ax.imshow(plt.get_cmap("inferno")(level)[..., :3], extent=(-50, 50, -50, 50), origin="upper",
              interpolation="bilinear")
    halo = [pe.withStroke(linewidth=1.6, foreground="black")]
    for key, name, centre in vehicles:
        ax.add_patch(Circle(centre, RADIUS, fill=False, edgecolor=ACCENT, linewidth=1.0))
        (lx, ly), ha = LABELS[key]
        edge = centre + RADIUS * (np.array([lx, ly]) - centre) / np.hypot(*(np.array([lx, ly]) - centre))
        ax.plot([edge[0], lx], [edge[1], ly], color=ACCENT, linewidth=0.6)
        ax.text(lx + (0.6 if ha == "left" else -0.6), ly, name, ha=ha, va="center", fontsize=6.2, color="white",
                path_effects=halo, linespacing=1.05)
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ticks = [-50, -25, 0, 25, 50]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(length=2, pad=1.5)
    ax.set_xlabel("$x$ (m)", labelpad=1)
    ax.set_ylabel("$y$ (m)", labelpad=1)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / "gotcha_scene_sar"
    for ext in args.formats.split(","):
        fig.savefig(f"{stem}.{ext}", dpi=400)
        if ext == "svg":
            retarget_svg_family(f"{stem}.svg")
    print("wrote", stem, f"{W:.2f} x {H:.2f} in", {key: centre.round(2).tolist() for key, _, centre in vehicles})


if __name__ == "__main__":
    main()
