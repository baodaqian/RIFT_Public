#!/usr/bin/env python3
"""Appendix schematic: RIFT on GOTCHA (fig:gotcha-rift-pipeline), in Fig. 1's visual language.

Content agreed with the writing agent (2026-09-25) against the paper text (sec/3 section 3.4, App. 8, App. 11,
App. 13); the paper's words, no run or arm names, no scoring machinery. Draft 2 (user, 2026-09-25): the scene is the
published staging photo of the GOTCHA parking lot with red-orange cubes on the measured cars, and (b) is drawings with
few words (the recipe itself is in the text).

(a) The measured scene and its viewpoints: the viewing sphere of Fig. 1(a), but the only viewpoints are the eight
    circular passes in a narrow elevation band (true elevations from the antenna phase centres, sphere not to scale);
    TRAIN sector directions as dots (sealed test sectors leave gaps); an airborne platform; at the centre the staging
    photo of Casteel et al. (2007, Fig. 2) with a red-orange cube (the region of interest) on each measured car
    (``--photo-cars``, the user's identification). Inset: pass x one-degree sector units in an azimuth window,
    training (gray), validation (open, pass 4), sealed test (blank columns across all passes).
(b) RIFT on GOTCHA as three drawings: backprojection initialization (measured pulses -> backprojection -> a coarse
    lattice),
    the fit of the measured responses (the data term of Fig. 1), and the coarse-to-fine enhancement (one cell -> 8 ->
    64 half-pitch children), which alternates with the fit.
(c) Coarse to fine on the Toyota Camry: plan-view maximum-intensity projections of the point energy of the one
    training run behind the Camry row, deposited at each stage's own lattice pitch (prepare_gotcha_pipeline_
    schematic_data_pvc.py): the recomputed backprojection initialization, the states before the densifications after
    epochs 10 and 16, and the selected epoch-40 checkpoint.

Palette and type as Fig. 1: RIFT red-orange, structure in ink and gray, fields on inferno, Times text.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts_pvc.paper_figure_font import retarget_svg_family, use_paper_font  # noqa: E402
from scripts_pvc.plot_rift_schematic_pvc import ACCENT, ACCENT_TINT, BASE, GRID, INK, LIGHT, MUTED  # noqa: E402

DATA = Path("/scratch/user/u.db364833/RIFT_runs/paper_figure_drafts_20260924/gotcha_pipeline_draft/schematic_data.npz")
PHOTO = Path("/scratch/group/p.cis261724.000/RIFT_pvc_runs/gotcha_new_targets_20260924/casteel2007_fig2_staging_photo.jpg")
# the nine staged cars of the photo, numbered left to right: bounding boxes in photo pixels (x0, y0, x1, y1)
PHOTO_CAR_BOXES = {1: (18, 68, 95, 102), 2: (122, 55, 197, 90), 3: (330, 74, 422, 115), 4: (467, 57, 556, 92),
                   5: (515, 96, 612, 137), 6: (588, 30, 668, 68), 7: (655, 75, 747, 112), 8: (768, 52, 860, 86),
                   9: (1047, 66, 1143, 104)}
PHOTO_CROP = (280, 880, 0, 200)             # photo pixels shown in (a): x0, x1, y0, y1 (the staged cars and the lot)
PERCENTILE = 99.5                           # panel (c) display scale (see mip_rgb)
WINDOW = (21, 40)                           # inset sectors: holds validation units and sealed test sectors
VIEW = (-35.0, 22.0)                        # the oblique camera of (a) and of the glyphs in (b)


def camera(azimuth_deg, elevation_deg):
    """Orthographic camera (x forward, y left, z up): (right, up, towards-camera) unit vectors."""
    az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
    towards = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    right = np.cross([0.0, 0.0, 1.0], towards)
    right /= np.linalg.norm(right)
    return right, np.cross(towards, right), towards


def project(points, view, scale=1.0, offset=(0.0, 0.0)):
    right, up, _ = view
    points = np.atleast_2d(points)
    return np.stack([points @ right * scale + offset[0], points @ up * scale + offset[1]], -1)


def plane_glyph(ax, centre, size, heading_deg, zorder=7):
    """A small airborne platform (top-view silhouette), nose along ``heading_deg`` on the page."""
    half = [(1.0, 0.0), (0.82, 0.08), (0.18, 0.08), (-0.22, 0.88), (-0.36, 0.88), (-0.18, 0.08), (-0.72, 0.07),
            (-0.92, 0.36), (-1.04, 0.36), (-0.97, 0.04), (-1.02, 0.0)]
    outline = np.array(half + [(x, -y) for x, y in half[::-1][1:-1]])
    c, s = np.cos(np.radians(heading_deg)), np.sin(np.radians(heading_deg))
    rotated = np.stack([outline[:, 0] * c - outline[:, 1] * s, outline[:, 0] * s + outline[:, 1] * c], 1)
    ax.add_patch(Polygon(rotated * size + centre, closed=True, facecolor=INK, edgecolor=INK, linewidth=0.3,
                         zorder=zorder))


def lattice_icon(ax, centre, size, n, view, point_scale=1.0, weights=None, zorder=5):
    """A cube of edge ``size`` (page units) split into n^3 cells: grid lines on its three visible faces, near edges
    solid, and a red-orange point at every cell centre (far points fainter; area from ``weights``)."""
    right, up, towards = view
    half = 0.5

    def page(p):
        return np.stack([np.atleast_2d(p) @ right, np.atleast_2d(p) @ up], -1) * size + centre

    ticks = np.linspace(-half, half, n + 1)
    for axis in range(3):
        sign = 1.0 if towards[axis] > 0 else -1.0                        # the visible face on this axis
        a, b = [i for i in range(3) if i != axis]
        for t in ticks:
            for vary, fixed in ((a, b), (b, a)):
                seg = np.zeros((2, 3))
                seg[:, axis] = sign * half
                seg[:, fixed] = t
                seg[:, vary] = [-half, half]
                edge = abs(abs(t) - half) < 1e-9
                ax.plot(*page(seg).T, color=INK if edge else GRID, linewidth=0.55 if edge else 0.35,
                        zorder=zorder + (1 if edge else 0))
    centres = (np.arange(n) + 0.5) / n - 0.5
    cells = np.array([[x, y, z] for x in centres for y in centres for z in centres])
    order = np.argsort(cells @ towards)
    depth = cells @ towards
    near = (depth - depth.min()) / (np.ptp(depth) + 1e-12)
    w = np.ones(len(cells)) if weights is None else np.asarray(weights, float)
    xy = page(cells)
    ax.scatter(xy[order, 0], xy[order, 1], s=point_scale * w[order], linewidths=0, zorder=zorder + 2,
               color=[(0.867, 0.318, 0.227, 0.35 + 0.65 * a) for a in near[order]])


def trace(x, phase, freq=3.2):
    """A band-limited pulse-like wiggle on [0, 1] (drawing only)."""
    envelope = np.exp(-((x - 0.5) / 0.30) ** 2)
    return envelope * np.cos(2 * np.pi * freq * x + phase)


def mip_rgb(plan):
    """sqrt(energy) over its PERCENTILE-th percentile among nonzero cells, clipped to 1, on inferno (black at zero).

    Not the view figures' min-max scale: at fine pitch a few hot cells set the maximum and the rest turns black."""
    magnitude = np.sqrt(np.clip(plan, 0, None))
    reference = np.percentile(magnitude[magnitude > 0], PERCENTILE)
    return plt.get_cmap("inferno")(np.clip(magnitude / reference, 0, 1))[..., :3]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--formats", default="pdf,svg,png")
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--photo-cars", type=int, nargs="*", default=[],
                   help="photo car numbers (left to right, PHOTO_CAR_BOXES) that get a cube: the measured vehicles")
    args = p.parse_args()
    use_paper_font()
    plt.rcParams["mathtext.cal"] = "cmsy10"
    plt.rcParams.update({"font.size": 7, "axes.edgecolor": MUTED, "axes.linewidth": 0.5,
                         "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 5.6, "ytick.labelsize": 5.6})
    data = np.load(args.data)
    meta = json.loads(str(data["meta"]))
    acquisition, stages = meta["acquisition"], meta["stages"]
    view = camera(*VIEW)

    W, H = 5.5, 2.55
    fig = plt.figure(figsize=(W, H))

    def axes_in(x0, y0, w, h, **kw):
        return fig.add_axes([x0 / W, y0 / H, w / W, h / H], **kw)

    def label(x, y, text, **kw):
        kw = dict(dict(ha="left", va="top", fontsize=7, color=INK), **kw)
        fig.text(x / W, y / H, text, **kw)

    top = H - 0.04
    AX0, BX0, BW, CX0 = 0.05, 2.20, 1.54, 3.86
    label(AX0, top, "(a) measured scene and viewpoints", fontsize=7.5, fontweight="bold")
    label(BX0, top, "(b) RIFT on GOTCHA", fontsize=7.5, fontweight="bold")
    label(CX0, top, "(c) coarse to fine: Toyota Camry", fontsize=7.5, fontweight="bold")

    # ---- (a) sphere with the pass band, platform and the photographed scene; inset of units ---------------------
    passes = {int(k): v for k, v in acquisition["passes"].items()}
    elevations = np.array([passes[k]["elevation_deg"][1] for k in sorted(passes)])
    R, scx, scy = 0.76, AX0 + 1.03, 1.60                              # sphere radius and centre (in)
    ax = axes_in(scx - 1.08 * R, scy - 1.08 * R, 2.16 * R, 2.16 * R)
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Circle((0, 0), 1.0, fill=False, edgecolor=GRID, linewidth=0.6))
    t = np.radians(np.linspace(0, 360, 361))
    equator = project(np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], 1), view)
    ax.plot(*equator.T, color=GRID, linewidth=0.4, linestyle=(0, (2, 2)))
    selected = sorted(acquisition["selected_sector_ids"])
    for el in elevations:                                             # eight rings, overlapping at this scale
        az = np.radians(np.array(selected) - 0.5)
        ring = np.stack([np.cos(np.radians(el)) * np.cos(az), np.cos(np.radians(el)) * np.sin(az),
                         np.full_like(az, np.sin(np.radians(el)))], 1)
        front = ring @ view[2] >= 0
        xy = project(ring, view)
        ax.scatter(*xy[front].T, s=0.35, color=BASE, linewidths=0, zorder=3)
        ax.scatter(*xy[~front].T, s=0.35, color=GRID, linewidths=0, zorder=1)
    cx0, cx1, cy0, cy1 = PHOTO_CROP
    photo = mpimg.imread(PHOTO)[cy0:cy1, cx0:cx1]
    ph_px, pw_px = photo.shape[:2]
    pw = 1.84                                                         # photo width in sphere units
    ph = pw * ph_px / pw_px
    px0, py1 = -pw / 2, -0.12 + ph / 2                                 # left, top edge
    ax.imshow(photo, extent=(px0, px0 + pw, py1 - ph, py1), zorder=4, interpolation="bilinear")
    ax.add_patch(Rectangle((px0, py1 - ph), pw, ph, fill=False, edgecolor=MUTED, linewidth=0.4, zorder=5))

    def photo_xy(u, v):                                               # full-photo pixels -> sphere units
        return px0 + (u - cx0) / pw_px * pw, py1 - (v - cy0) / ph_px * ph

    for number in args.photo_cars:                                    # a cube on each measured car
        x0, y0, x1, y1 = PHOTO_CAR_BOXES[number]
        (a0, b0), (a1, b1) = photo_xy(x0, y1), photo_xy(x1, y0)        # front face: lower-left, upper-right
        dx, dy = 0.16 * (a1 - a0), 0.32 * (b1 - b0)                    # oblique depth offset
        front = np.array([(a0, b0), (a1, b0), (a1, b1), (a0, b1)])
        back = front + [dx, dy]
        for i in range(4):
            ax.plot(*np.stack([front[i], front[(i + 1) % 4]]).T, color=ACCENT, linewidth=0.7, zorder=6)
            ax.plot(*np.stack([back[i], back[(i + 1) % 4]]).T, color=ACCENT, linewidth=0.5, zorder=6)
            ax.plot(*np.stack([front[i], back[i]]).T, color=ACCENT, linewidth=0.5, zorder=6)
    el_mid = np.radians(elevations.mean())
    az_platform = np.radians(-75.0)
    platform = np.array([np.cos(el_mid) * np.cos(az_platform), np.cos(el_mid) * np.sin(az_platform), np.sin(el_mid)])
    pxy = project(platform, view)[0]
    tangent = project(platform + 0.05 * np.array([-np.sin(az_platform), np.cos(az_platform), 0.0]), view)[0] - pxy
    plane_glyph(ax, pxy, 0.075, np.degrees(np.arctan2(tangent[1], tangent[0])))
    target = np.array([0.0, py1 + 0.02])
    direction = (target - pxy) / np.hypot(*(target - pxy))
    ax.plot(*np.stack([pxy + 0.10 * direction, target]).T, color=MUTED, linewidth=0.6, linestyle=(0, (3, 2)),
            zorder=5)
    low = min(passes[k]["elevation_deg"][0] for k in passes)          # over every pulse of every pass
    high = max(passes[k]["elevation_deg"][2] for k in passes)
    label(AX0, top - 0.17, f"8 passes,\nelevation\n{low:.1f}–{high:.1f}°", fontsize=5.6, color=MUTED, linespacing=1.1)
    if args.photo_cars:
        label(AX0 + 1.03 + 0.72 * R, scy - 0.10 * R - 0.5 * ph * R - 0.02, "region of interest $\\Omega$",
              ha="right", fontsize=5.6, color=ACCENT)

    held, sealed = set(acquisition["heldout_sector_ids"]), set(acquisition["sealed_test_sector_ids"])
    order = sorted(passes, key=lambda k: passes[k]["elevation_deg"][1], reverse=True)
    ix = axes_in(0.32, 0.42, 1.60, 0.34)
    for row, pass_id in enumerate(order[::-1]):
        for sector in range(WINDOW[0], WINDOW[1] + 1):
            if sector in sealed:
                continue
            if pass_id == acquisition["heldout_pass"] and sector in held:
                ix.scatter(sector, row, s=5.5, facecolors="white", edgecolors=INK, linewidths=0.5, zorder=3)
            else:
                ix.scatter(sector, row, s=2.4, color=BASE, linewidths=0, zorder=2)
    for sector in range(WINDOW[0], WINDOW[1] + 1):
        if sector in sealed:
            ix.add_patch(Rectangle((sector - 0.45, -0.6), 0.9, len(order) + 0.2, facecolor=LIGHT, edgecolor="none",
                                   zorder=1))
    rows = order[::-1]
    ix.set_xlim(WINDOW[0] - 0.7, WINDOW[1] + 0.7)
    ix.set_ylim(-0.8, len(rows) - 0.2)
    shown = [r for r, k in enumerate(rows) if r in (0, len(rows) - 1) or k == acquisition["heldout_pass"]]
    ix.set_yticks(shown)
    ix.set_yticklabels([str(rows[r]) for r in shown])
    ix.set_xticks([])
    ix.tick_params(length=0, pad=1.5, labelsize=5.0)
    ix.set_ylabel("pass", fontsize=5.4, labelpad=1)
    ix.set_xlabel("one-degree azimuth sectors", fontsize=5.4, labelpad=1)
    for side in ("top", "right", "left", "bottom"):
        ix.spines[side].set_visible(False)
    lg = axes_in(0.32, 0.78, 1.60, 0.07)
    lg.axis("off")
    lg.set_xlim(0, 1.60)
    lg.set_ylim(0, 1)
    lg.scatter([0.03], [0.5], s=2.4, color=BASE, linewidths=0)
    lg.text(0.07, 0.5, "training", va="center", fontsize=5.4, color=MUTED)
    lg.scatter([0.46], [0.5], s=5.5, facecolors="white", edgecolors=INK, linewidths=0.5)
    lg.text(0.50, 0.5, "validation", va="center", fontsize=5.4, color=MUTED)
    lg.add_patch(Rectangle((0.95, 0.12), 0.05, 0.76, facecolor=LIGHT, edgecolor="none"))
    lg.text(1.03, 0.5, "sealed test", va="center", fontsize=5.4, color=MUTED)
    label(AX0, 0.25, "measured HH responses $\\mathbf{S}_v$ of (pass, sector) units\n"
          "$v \\in \\mathcal{V}_{\\mathrm{tr}}$, restricted to the region of interest", fontsize=6.2,
          linespacing=1.12)

    # ---- (b) RIFT on GOTCHA: three drawings ---------------------------------------------------------------------
    by0, bh = 0.05, top - 0.20 - 0.05
    fig.patches.append(FancyBboxPatch((BX0, by0), BW, bh, transform=fig.dpi_scale_trans, zorder=-1,
                                      boxstyle="round,pad=0.02,rounding_size=0.08", facecolor=ACCENT_TINT,
                                      edgecolor=ACCENT, linewidth=0.9))
    bx = axes_in(BX0, by0, BW, bh)                                    # data units = inches inside the box
    bx.set_xlim(0, BW)
    bx.set_ylim(0, bh)
    bx.axis("off")
    bx.patch.set_alpha(0)

    def badge(x, y, number):
        bx.add_patch(Circle((x, y), 0.052, facecolor=ACCENT, edgecolor="none", zorder=8))
        bx.text(x, y, str(number), ha="center", va="center", fontsize=5.8, color="white", fontweight="bold", zorder=9)

    def head(y, number, text):
        badge(0.12, y - 0.055, number)
        bx.text(0.22, y, text, ha="left", va="top", fontsize=6.5, fontweight="bold", color=INK, linespacing=1.1)

    arrow = dict(arrowstyle="-|>", mutation_scale=6, linewidth=0.7, color=ACCENT, shrinkA=0, shrinkB=0)
    # 1: measured pulses -> backprojection -> coarse lattice
    y1 = bh - 0.07
    head(y1, 1, "backprojection initialization")
    xs = np.linspace(0, 1, 120)
    for k in range(3):
        base = y1 - 0.30 - 0.085 * k
        bx.plot(0.17 + 0.31 * xs, base + 0.035 * trace(xs, 1.3 * k, 2.6), color=BASE, linewidth=0.6, zorder=5)
    bx.text(0.17, y1 - 0.58, "measured pulses", ha="left", va="top", fontsize=5.4, color=MUTED)
    bx.add_patch(FancyArrowPatch((0.53, y1 - 0.385), (0.96, y1 - 0.385), **arrow))
    bx.text(0.745, y1 - 0.365, "backprojection", ha="center", va="bottom", fontsize=5.2, color=MUTED, style="italic")
    cells = (np.arange(3) + 0.5) / 3 - 0.5
    blob = np.array([np.exp(-((x - 0.1) ** 2 + y ** 2 + (z + 0.1) ** 2) / 0.12) for x in cells for y in cells
                     for z in cells])
    lattice_icon(bx, np.array([1.20, y1 - 0.39]), 0.30, 3, view, point_scale=14.0, weights=0.25 + blob)
    # 2: fit the measured responses
    y2 = y1 - 0.72
    head(y2, 2, "fit the measured responses")
    bx.text(BW / 2 + 0.05, y2 - 0.25, "$\\min\\;\\sum_{v \\in \\mathcal{V}_{\\mathrm{tr}}} \\ell_v(\\widehat{\\mathbf{S}}_v,"
            "\\,\\mathbf{S}_v)$", ha="center", va="center", fontsize=7.2, color=INK)
    xs = np.linspace(0, 1, 160)
    wave_y = y2 - 0.52
    bx.plot(0.30 + 0.95 * xs, wave_y + 0.07 * trace(xs, 0.4, 4.0), color=ACCENT, linewidth=0.8, zorder=5)
    sample = np.linspace(0.04, 0.96, 26)
    bx.scatter(0.30 + 0.95 * sample, wave_y + 0.07 * trace(sample, 0.4, 4.0) * (1 + 0.12 * np.sin(9 * sample)),
               s=2.6, color=BASE, linewidths=0, zorder=6)
    bx.text(0.26, wave_y + 0.045, "$\\widehat{\\mathbf{S}}_v$", ha="right", va="center", fontsize=6.2, color=ACCENT)
    bx.text(0.26, wave_y - 0.045, "$\\mathbf{S}_v$", ha="right", va="center", fontsize=6.2, color=BASE)
    # 3: coarse-to-fine enhancement: one cell -> 8 -> 64 half-pitch children, alternating with the fit
    y3 = y2 - 0.70
    head(y3, 3, "coarse-to-fine enhancement\n(densification)")
    icon_y = y3 - 0.44
    for x, n, scale in ((0.30, 1, 16.0), (0.76, 2, 7.0), (1.22, 4, 2.2)):
        lattice_icon(bx, np.array([x, icon_y]), 0.26, n, view, point_scale=scale)
    for xa, xb in ((0.47, 0.58), (0.93, 1.04)):
        bx.add_patch(FancyArrowPatch((xa, icon_y), (xb, icon_y), **arrow))
        bx.text((xa + xb) / 2, icon_y + 0.035, "×8", ha="center", va="bottom", fontsize=5.4, color=ACCENT)
    bx.text(BW / 2, icon_y - 0.19, "after epochs 4, 10, 16", ha="center", va="top", fontsize=5.6, color=MUTED)
    # flow: 1 -> 2 -> 3 down the badges, and 3 -> 2 on the right (the fit resumes after each densification)
    for y_from, y_to in ((y1 - 0.13, y2 + 0.005), (y2 - 0.13, y3 + 0.005)):
        bx.add_patch(FancyArrowPatch((0.12, y_from), (0.12, y_to), **dict(arrow, linewidth=0.5)))
    bx.add_patch(FancyArrowPatch((BW - 0.09, y3 - 0.10), (BW - 0.09, y2 - 0.40), **arrow))
    bx.text(BW - 0.12, (y3 + y2) / 2 - 0.25, "repeat", ha="right", va="center", fontsize=5.4, color=ACCENT,
            style="italic", rotation=90)

    # ---- (c) coarse to fine ------------------------------------------------------------------------------------
    rows_c = [("epoch0", "epoch 0, initialization"), ("epoch10", "epoch 10"), ("epoch16", "epoch 16"),
              ("epoch40", "epoch 40, selected")]
    events = ["densify after epoch 4", "densify after epoch 10", "densify after epoch 16"]
    pitch_text = {0.125: "0.125 m", 0.0625: "0.0625 m", 0.03125: "0.031 m", 0.015625: "0.0156 m"}
    cy_top, cy_bottom, gap = top - 0.22, 0.05, 0.14
    tile_h = (cy_top - cy_bottom - 3 * gap) / 4
    tile_w = 2 * tile_h                                               # the 6 m x 3 m plan window
    tile_arrow = dict(arrowstyle="-|>", mutation_scale=5, linewidth=0.7, color=ACCENT)
    for i, (key, name) in enumerate(rows_c):
        y0 = cy_top - (i + 1) * tile_h - i * gap
        t_ax = axes_in(CX0, y0, tile_w, tile_h)
        plan = data[f"plan_{key}"]                                     # [x, y] -> image rows y (up), columns x
        t_ax.imshow(mip_rgb(plan).transpose(1, 0, 2), origin="lower", extent=(-3, 3, -1.5, 1.5),
                    interpolation="nearest", aspect="auto")
        t_ax.set_xticks([])
        t_ax.set_yticks([])
        for spine in t_ax.spines.values():
            spine.set_edgecolor("#252525")
        s = stages[key]
        title, _, note = name.partition(", ")
        lx, ly = CX0 + tile_w + 0.06, y0 + tile_h - 0.005
        label(lx, ly, title, fontsize=6.2, fontweight="bold")
        details = ([note] if note else []) + [f"pitch {pitch_text[s['pitch_m']]}", f"{s['active']:,} points"]
        label(lx, ly - 0.105, "\n".join(details), fontsize=5.7, color=MUTED, linespacing=1.15)
        if i < 3:
            ay_top, ay_bot = y0 - 0.008, y0 - gap + 0.008
            fig.patches.append(FancyArrowPatch((CX0 + 0.10, ay_top), (CX0 + 0.10, ay_bot),
                                               transform=fig.dpi_scale_trans, **tile_arrow))
            label(CX0 + 0.17, (ay_top + ay_bot) / 2, events[i], va="center", fontsize=5.3, color=ACCENT,
                  style="italic")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / "gotcha_rift_pipeline_draft"
    for ext in args.formats.split(","):
        fig.savefig(f"{stem}.{ext}", dpi=400)
        if ext == "svg":
            retarget_svg_family(f"{stem}.svg")
    print("wrote", stem, "| cubes on photo cars", args.photo_cars or "none")


if __name__ == "__main__":
    main()
