#!/usr/bin/env python3
"""Per-view signal errors on the RIFT-dataset view sphere (TRAIN + VALIDATION views), appendix figures.

Every radar viewpoint of an object lies on a sphere around it (10 m standoff). Each TRAIN and VALIDATION view
gets one number: its complex RelMSE sum_f |S_pred(f) - S(f)|^2 / sum_f |S(f)|^2 over the 600 frequencies of the
1 Tx x 1 Rx response, the per-view form of the paper's Complex RelMSE. The sphere is coloured by the nearest
TRAIN or VALIDATION view (a spherical Voronoi fill; reserved-test and unused directions take their nearest
neighbour's colour, no value is interpolated). One log colour scale is shared by every panel.

Methods with a complex prediction: RIFT, SpINR-style, GeRaF and Sugavanam-Ertin Stage 1. Values come from
``scripts_pvc/export_rift_dataset_sphere_response_pvc.py`` (<root>/<object>/<method>.npz). ``--eval-cache``
instead reads the final evaluation's VALIDATION-only caches (layout checks while the TRAIN readout runs).

``--observable power`` draws the per-view common matched-range power RelMSE instead (normalized dB over the view's
ROI range bins, the metric of the per-object signal table's second block; <method>_power.npz from the export's power
mode) for every method with a range-power prediction: the four above plus Radar Fields and RadarSplat (scored on
the ROI bins inside its crop, marked with a section sign as in the tables).

Layouts (--layout): ``globes`` (two antipodal orthographic views per method) and ``mollweide`` (one equal-area
map per method, longitude = azimuth about the object's up axis from its nose, latitude = elevation).
Text in the paper's Times (scripts_pvc/paper_figure_font.py).
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts_pvc.paper_figure_font import retarget_svg_family, use_paper_font  # noqa: E402

# (dataset object id, final-eval label, row label, nose axis, up axis) -- the frames of the paper's view
# figures (run_rift_dataset_six_method_postflight_pvc.SCENE_VIEWS): A320 and X-59 nose -x, up +z; the
# other four nose +z, up +y.
OBJECTS = [
    ("b787", "b787", "B787", (0, 0, 1), (0, 1, 0)),
    ("airliner_a320", "a320", "A320", (-1, 0, 0), (0, 0, 1)),
    ("supersonic_x59", "x59", "X-59", (-1, 0, 0), (0, 0, 1)),
    ("firetruck", "firetruck", "Fire truck", (0, 0, 1), (0, 1, 0)),
    ("race_car", "race_car", "Race car", (0, 0, 1), (0, 1, 0)),
    ("loader", "loader", "Loader", (0, 0, 1), (0, 1, 0)),
]
METHODS = [("rift", "RIFT (ours)"), ("spinr", "SpINR-style"), ("geraf", "GeRaF"),
           ("sugavanam_ertin", "Sugavanam–Ertin, Stage 1")]
# per observable: methods (key, column header), per-object file, per-view field, colour range (%), bar label, stem
OBSERVABLES = {
    "complex": dict(methods=METHODS, file="{method}.npz", field="coherent_rel_mse", range=(0.1, 1000.0),
                    label="Per-view complex RelMSE (%)", stem="rift_dataset_sphere_error"),
    "power": dict(methods=[("rift", "RIFT (ours)"), ("spinr", "SpINR-style"), ("geraf", "GeRaF"),
                           ("sugavanam_ertin", "Sugavanam–Ertin,\nStage 1"), ("radar_fields", "Radar Fields"),
                           ("radarsplat", "RadarSplat§")],
                  file="{method}_power.npz", field="range_power_rel_mse", range=(0.01, 1000.0),
                  label="Per-view common range-power RelMSE (%)", stem="rift_dataset_sphere_power_error"),
}

# The paper's figure scale (user, 09-24): inferno, as in the MIP view figures -- brighter = larger (here: more error).
CMAP = matplotlib.colormaps["inferno"]
NORM = LogNorm(vmin=0.1, vmax=1000.0)      # percent
INK, MUTED, EDGE = "#0b0b0b", "#52514e", "#8a8984"
GRID_LINE = "#bdbcb6"                      # reads on inferno's black and its yellow alike
RASTER_DPI = 400      # PNG, and the embedded map/globe images inside the PDF and SVG

# Globe cameras in the object frame (nose, left, up): front-left from above, and its antipode.
CAM_AZ, CAM_EL = np.deg2rad(35.0), np.deg2rad(25.0)


def style():
    use_paper_font()
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
                         "axes.edgecolor": MUTED, "axes.linewidth": 0.5, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.labelcolor": INK})


def frame(nose, up):
    f, u = np.asarray(nose, float), np.asarray(up, float)
    return f, np.cross(u, f), u          # nose, left, up (right-handed)


def load(args, obj_id, label, method):
    obs = OBSERVABLES[args.observable]
    if args.eval_cache:
        with np.load(Path(args.eval_cache) / f"{label}_{method}_val_per_view.npz") as z:
            return z["viewpoint_positions"], 100 * z[obs["field"]]
    with np.load(Path(args.root) / obj_id / obs["file"].format(method=method)) as z:
        return z["viewpoint_positions"], 100 * z[obs["field"]]


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def globe_rgba(tree, values, cam, up, npx):
    """Orthographic image of the visible hemisphere; each pixel takes its nearest view's colour."""
    c = unit(cam)
    u = unit(up - np.dot(up, c) * c)
    r = np.cross(u, c)
    s = np.linspace(-1, 1, npx)
    x, y = np.meshgrid(s, -s)
    inside = x * x + y * y <= 1.0
    z = np.sqrt(np.clip(1 - x * x - y * y, 0, None))
    p = x[..., None] * r + y[..., None] * u + z[..., None] * c
    _, idx = tree.query(p[inside])
    rgba = np.zeros((npx, npx, 4))
    rgba[inside] = CMAP(NORM(np.clip(values[idx], NORM.vmin, NORM.vmax)))
    return rgba, (r, u, c)


def graticule(ax, basis, f, left, up):
    """Latitude circles every 30 deg and meridians every 30 deg about the object's up axis (visible parts)."""
    r, u, c = basis
    t = np.linspace(0, 2 * np.pi, 361)
    lines = []
    for lat in np.deg2rad([-60, -30, 0, 30, 60]):
        lines.append((np.cos(lat) * (np.cos(t)[:, None] * f + np.sin(t)[:, None] * left) + np.sin(lat) * up,
                      0.55 if lat == 0 else 0.3))
    h = np.linspace(-np.pi / 2, np.pi / 2, 181)
    for lon in np.deg2rad(np.arange(0, 360, 30)):
        d = np.cos(lon) * f + np.sin(lon) * left
        lines.append((np.cos(h)[:, None] * d + np.sin(h)[:, None] * up, 0.55 if lon == 0 else 0.3))
    for pts, lw in lines:
        vis = pts @ c > 0
        xs, ys = np.where(vis, pts @ r, np.nan), np.where(vis, pts @ u, np.nan)
        ax.plot(xs, ys, color=GRID_LINE, alpha=0.45, linewidth=lw, solid_capstyle="round")
    halo = [patheffects.withStroke(linewidth=1.6, foreground=INK)]
    for vec, mark in ((f, "o"), (up, "+")):            # nose (open circle) and up pole (plus), if visible
        if vec @ c > 0:
            ax.plot(vec @ r, vec @ u, marker=mark, markersize=3.2 if mark == "o" else 4.5, markerfacecolor="none",
                    markeredgecolor="white", markeredgewidth=0.7, path_effects=halo)


def draw_globe(ax, positions, values, nose, up_axis, side, npx):
    f, left, up = frame(nose, up_axis)
    d = np.cos(CAM_EL) * (np.cos(CAM_AZ) * f + np.sin(CAM_AZ) * left) + np.sin(CAM_EL) * up
    cam = d if side == 0 else -d
    tree = cKDTree(unit(positions))
    rgba, basis = globe_rgba(tree, values, cam, up, npx)
    ax.imshow(rgba, extent=(-1, 1, -1, 1), interpolation="nearest", zorder=1)
    graticule(ax, basis, f, left, up)
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, edgecolor=EDGE, linewidth=0.5, zorder=3))
    ax.set_xlim(-1.04, 1.04)
    ax.set_ylim(-1.04, 1.04)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_mollweide(ax, positions, values, nose, up_axis, n_lon=360, n_lat=180):
    f, left, up = frame(nose, up_axis)
    lon = np.linspace(-np.pi, np.pi, n_lon + 1)
    lat = np.linspace(-np.pi / 2, np.pi / 2, n_lat + 1)
    lc, tc = 0.5 * (lon[1:] + lon[:-1]), 0.5 * (lat[1:] + lat[:-1])
    L, T = np.meshgrid(lc, tc)
    p = (np.cos(T) * np.cos(L))[..., None] * f + (np.cos(T) * np.sin(L))[..., None] * left + np.sin(T)[..., None] * up
    _, idx = cKDTree(unit(positions)).query(p.reshape(-1, 3))
    grid = np.clip(values[idx].reshape(L.shape), NORM.vmin, NORM.vmax)
    ax.pcolormesh(lon, lat, grid, cmap=CMAP, norm=NORM, shading="flat", rasterized=True)
    ax.set_xticks(np.deg2rad([-120, -60, 0, 60, 120]))
    ax.set_yticks(np.deg2rad([-60, -30, 0, 30, 60]))
    ax.tick_params(labelbottom=False, labelleft=False, length=0)
    ax.grid(True, color=GRID_LINE, alpha=0.45, linewidth=0.3)


def colorbar(fig, rect, label):
    cax = fig.add_axes(rect)
    sm = matplotlib.cm.ScalarMappable(norm=NORM, cmap=CMAP)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="both", extendfrac=0.04)
    decades = 10.0 ** np.arange(np.ceil(np.log10(NORM.vmin)), np.floor(np.log10(NORM.vmax)) + 1)
    cb.set_ticks(decades)
    cb.set_ticklabels([f"{d:g}" for d in decades])
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(length=2, width=0.4, pad=1.5)
    cb.set_label(label, labelpad=2)


def figure_globes(args, objects, out):
    width, left_pad, gap = 5.5, 0.52, 0.06
    cell = (width - left_pad - 0.08 - 3 * gap) / 8
    head, foot = 0.34, 0.52
    height = head + cell * len(objects) + foot
    obs = OBSERVABLES[args.observable]
    fig = plt.figure(figsize=(width, height))
    for j, (_, mlabel) in enumerate(obs["methods"]):
        x0 = left_pad + j * (2 * cell + gap)
        fig.text((x0 + cell) / width, 1 - 0.1 / height, mlabel, ha="center", va="top", fontsize=7.5, color=INK)
        for side, tag in enumerate(("front, above", "rear, below")):
            fig.text((x0 + (side + 0.5) * cell) / width, 1 - 0.24 / height, tag, ha="center", va="top",
                     fontsize=6, color=MUTED)
    for i, (obj_id, label, rlabel, nose, up_axis) in enumerate(objects):
        y0 = height - head - (i + 1) * cell
        fig.text(0.06 / width, (y0 + cell / 2) / height, rlabel, ha="left", va="center", fontsize=7, color=INK)
        for j, (method, _) in enumerate(obs["methods"]):
            positions, values = load(args, obj_id, label, method)
            for side in (0, 1):
                x0 = left_pad + j * (2 * cell + gap) + side * cell
                ax = fig.add_axes([x0 / width, y0 / height, cell / width, cell / height])
                draw_globe(ax, positions, values, nose, up_axis, side, args.npx)
    colorbar(fig, [(width / 2 - 1.3) / width, 0.3 / height, 2.6 / width, 0.07 / height], obs["label"])
    save(fig, out / f"{obs['stem']}_globes{args.suffix}", args.formats)


def figure_mollweide(args, objects, out):
    obs = OBSERVABLES[args.observable]
    methods = obs["methods"]
    width, left_pad, gap = 5.5, 0.52, 0.06
    cw = (width - left_pad - 0.08 - (len(methods) - 1) * gap) / len(methods)
    ch = cw / 2
    lines = max(label.count("\n") + 1 for _, label in methods)
    head, foot, vgap = 0.10 + 0.12 * lines, 0.52, 0.05
    height = head + len(objects) * (ch + vgap) + foot
    fig = plt.figure(figsize=(width, height))
    for j, (_, mlabel) in enumerate(methods):
        fig.text((left_pad + j * (cw + gap) + cw / 2) / width, 1 - 0.08 / height, mlabel, ha="center", va="top",
                 fontsize=7.5 if len(methods) <= 4 else 7, color=INK, linespacing=1.1)
    for i, (obj_id, label, rlabel, nose, up_axis) in enumerate(objects):
        y0 = height - head - (i + 1) * (ch + vgap) + vgap
        fig.text(0.06 / width, (y0 + ch / 2) / height, rlabel, ha="left", va="center", fontsize=7, color=INK)
        for j, (method, _) in enumerate(methods):
            positions, values = load(args, obj_id, label, method)
            ax = fig.add_axes([(left_pad + j * (cw + gap)) / width, y0 / height, cw / width, ch / height],
                              projection="mollweide")
            draw_mollweide(ax, positions, values, nose, up_axis)
            for s in ax.spines.values():
                s.set_linewidth(0.5)
                s.set_edgecolor(EDGE)
    colorbar(fig, [(width / 2 - 1.3) / width, 0.3 / height, 2.6 / width, 0.07 / height], obs["label"])
    save(fig, out / f"{obs['stem']}_mollweide{args.suffix}", args.formats)


def save(fig, stem, formats):
    for ext in formats:
        fig.savefig(f"{stem}.{ext}", dpi=RASTER_DPI)
        if ext == "svg":
            retarget_svg_family(f"{stem}.svg")
    plt.close(fig)
    print("wrote", stem, "+".join(formats))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--root", help="sphere-response root with <object>/<method>.npz (TRAIN + VALIDATION)")
    src.add_argument("--eval-cache", help="final-eval signal/ dir: VALIDATION-only per-view caches (layout checks)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--observable", choices=tuple(OBSERVABLES), default="complex")
    p.add_argument("--vmin", type=float, help="colour range floor in percent (default per observable)")
    p.add_argument("--layout", default="globes,mollweide")
    p.add_argument("--objects", default=",".join(o[0] for o in OBJECTS))
    p.add_argument("--formats", default="pdf,svg,png")
    p.add_argument("--npx", type=int, default=240, help="pixels across one globe")
    p.add_argument("--suffix", default="")
    args = p.parse_args()
    args.formats = args.formats.split(",")
    if args.eval_cache and args.observable != "complex":
        p.error("--eval-cache holds the complex layout checks only")
    global NORM
    low, high = OBSERVABLES[args.observable]["range"]
    NORM = LogNorm(vmin=args.vmin or low, vmax=high)
    objects = [o for o in OBJECTS if o[0] in args.objects.split(",")]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    style()
    for layout in args.layout.split(","):
        {"globes": figure_globes, "mollweide": figure_mollweide}[layout](args, objects, out)
    if args.root:
        summary = {o[0]: json.loads((Path(args.root) / o[0] / "summary.json").read_text()) for o in objects
                   if (Path(args.root) / o[0] / "summary.json").exists()}
        # provenance next to the data, never in the manuscript's figures/ tree
        (Path(args.root) / f"sphere_{args.observable}_error_sources.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
