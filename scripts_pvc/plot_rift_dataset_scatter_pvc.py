#!/usr/bin/env python3
"""Per-object scatter plots of the RIFT-dataset results.

Panels, each drawn from one per-object table of the appendix:
  * signal:   the main block of the per-object signal table (reserved TEST) --
              Complex RelMSE (x) against common matched-range power RelMSE (y).
              Methods without a complex output sit in a separate strip.
  * geometry: the per-object geometry table (fixed t = 0.20, 48^3 lattice) --
              Chamfer against F1, and HD95 against IoU.

Figures (--figures):
  * combined: the paper figure (main results section) -- (a) signal on top, (b) Chamfer x F1 and
              (c) HD95 x IoU below, one shared legend; stem rift_dataset_performance_scatter.
  * signal, geometry: the two standalone drafts of 09-24 (preview only).

One faint marker per (method, object) and one solid marker per method at its unweighted six-object mean (the
value Table 1 reports). RIFT is the accent colour (inferno(0.60), shared with the MIP
view figures); baselines are gray,
told apart by marker shape and a legend. Every value is read from the numbers file
the tables are filled from (scripts_pvc/paper_rift_dataset_numbers_pvc.py).

Text is set in the paper's typeface (scripts_pvc/paper_figure_font.py: TeX Gyre Termes, the TeX Live Times).
Writes <stem>.pdf, <stem>.svg (text kept as text and named Times New Roman, for Figma) and, with --formats,
<stem>.png for previews. The manuscript tree takes only the PDF and the SVG.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts_pvc.paper_figure_font import retarget_svg_family, use_paper_font  # noqa: E402

OBJECTS = ["b787", "a320", "x59", "firetruck", "race_car", "loader"]

ACCENT = "#dd513a"   # RIFT: inferno(0.60), from the colour scale of the MIP view figures (user, 09-24)
BASE = "#6b6a66"     # baselines
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
BAND = "#efeeea"
SURFACE = "#ffffff"

# key: (label, marker, filled)
METHODS = {
    "rift": ("RIFT (ours)", "o", True),
    "spinr": ("SpINR-style", "s", True),
    "se": ("SE", "D", True),                      # Sugavanam–Ertin, abbreviated as in the paper (user, 09-25)
    "se_stage1": ("SE, Stage 1 only", "D", False),
    "backprojection": ("Backprojection", "X", True),
    "geraf": ("GeRaF", "^", True),
    "radar_fields": ("Radar Fields", "v", True),
    "radarsplat": ("RadarSplat", "p", True),
}


def style():
    use_paper_font()
    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "axes.labelcolor": INK,
    })


FAINT = 0.22          # per-object markers (user, 09-25): largely transparent behind the six-object mean
MEAN_SIZE = 40        # the six-object mean (Table 1's value): solid, a little larger


def scatter(ax, key, xs, ys, size=26, z=3, alpha=1.0):
    label, marker, filled = METHODS[key]
    colour = ACCENT if key == "rift" else BASE
    ax.scatter(xs, ys, s=size, marker=marker,
               facecolors=colour if filled else SURFACE, edgecolors=colour if not filled else SURFACE,
               linewidths=0.9 if not filled else 0.5, zorder=z + (1 if key == "rift" else 0), clip_on=False,
               alpha=alpha)


def with_mean(ax, key, xs, ys, mean_xy=None):
    """Per-object markers faint; the unweighted six-object mean (as Table 1 reports it) solid on top."""
    scatter(ax, key, xs, ys, alpha=FAINT)
    mx, my = mean_xy if mean_xy is not None else (float(np.mean(xs)), float(np.mean(ys)))
    scatter(ax, key, [mx], [my], size=MEAN_SIZE, z=5)


def legend_handles(keys):
    handles = []
    for key in keys:
        label, marker, filled = METHODS[key]
        colour = ACCENT if key == "rift" else BASE
        handles.append(Line2D([], [], linestyle="none", marker=marker, markersize=5,
                              markerfacecolor=colour if filled else SURFACE,
                              markeredgecolor=colour, markeredgewidth=0.8 if not filled else 0.0,
                              label=label))
    return handles


def grid(ax):
    ax.grid(True, which="major", color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# Legend order, filled column by column (matplotlib's order) in a 4 x 2 legend: RIFT over SpINR-style,
# the two Sugavanam-Ertin entries, Backprojection over GeRaF, and the two power-only methods.
LEGEND_KEYS = ["rift", "spinr", "se", "se_stage1", "backprojection", "geraf", "radar_fields", "radarsplat"]
SIGNAL_COMPLEX = ["rift", "spinr", "se", "geraf"]
SIGNAL_POWER_ONLY = ["radar_fields", "radarsplat"]
GEOMETRY_KEYS = ["rift", "spinr", "se_stage1", "backprojection", "se", "geraf", "radarsplat", "radar_fields"]
GEOMETRY_PANELS = {
    "chamfer_f1": ("chamfer", "f1", "Chamfer ($10^{-4}$ m$^2$)", "F1-score", (0.2, 400)),
    "hd95_iou": ("hd95", "iou", "HD95 (mm)", "IoU", (5, 300)),
}


def plain_log_ticks(axis):
    axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))


def better_notes(ax, x_better, y_better, y_at=0.5):
    """Inside the axes, just off each spine: "<- lower is better" or "higher is better ->" (on the vertical axis
    the arrow points down or up), centred on the horizontal axis and at `y_at` (axes fraction) on the vertical one.
    The notes clear the zero gridline of the geometry panels and sit on a small surface patch over the grid."""
    text = {"lower": "← lower is better", "higher": "higher is better →"}
    common = dict(xycoords="axes fraction", textcoords="offset points", fontsize=6.5, color=MUTED, zorder=2.5,
                  bbox=dict(boxstyle="square,pad=0.15", facecolor=SURFACE, edgecolor="none"))
    ax.annotate(text[x_better], xy=(0.5, 0.0), xytext=(0, 4.5), ha="center", va="bottom", **common)
    ax.annotate(text[y_better], xy=(0.0, y_at), xytext=(2.5, 0), ha="left", va="center", rotation=90, **common)


def panel_tag(ax, tag):
    ax.text(-0.02, 1.04, tag, transform=ax.transAxes, fontsize=8, fontweight="bold", ha="right", va="bottom")


def draw_signal(ax, sx, numbers):
    """Complex x common range-power RelMSE on `ax`; the power-only methods in the strip `sx` (sharey=ax)."""
    sig = numbers["signal"]
    floors = [numbers["floors"][o]["full_roi"] for o in OBJECTS]
    crop_floors = [numbers["floors"][o]["radarsplat_crop"] for o in OBJECTS]

    ax.axhspan(min(floors), max(floors), color=BAND, zorder=0.5, linewidth=0)
    sx.fill_between([-0.5, 0.5], min(floors), max(floors), color=BAND, zorder=0.5, linewidth=0)
    sx.fill_between([0.5, 1.5], min(crop_floors), max(crop_floors), color=BAND, zorder=0.5, linewidth=0)
    ax.text(0.22, max(floors) * 1.06, "TRAIN-mean constant (range over objects)", fontsize=6.5,
            color=MUTED, va="bottom", ha="left")

    ax.axvline(100.0, color=MUTED, linewidth=0.6, zorder=1)
    ax.text(100.0 * 0.92, 0.23, "predicting zero", rotation=90, fontsize=6.5, color=MUTED,
            ha="right", va="bottom")

    for key in SIGNAL_COMPLEX:
        xs = [sig[o][key]["complex"] for o in OBJECTS]
        ys = [sig[o][key]["power"] for o in OBJECTS]
        with_mean(ax, key, xs, ys)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.2, 300)
    ax.set_ylim(0.2, 100)
    ax.set_xlabel("Complex RelMSE (%)")
    ax.set_ylabel("Common range-power RelMSE (%)")
    plain_log_ticks(ax.xaxis)
    plain_log_ticks(ax.yaxis)
    grid(ax)
    better_notes(ax, "lower", "lower", y_at=0.40)       # centred below the floor band

    # strip: methods with no complex output, one column each, objects spread across it
    for col, key in enumerate(SIGNAL_POWER_ONLY):
        xs = [col + (i - 2.5) * 0.09 for i in range(len(OBJECTS))]
        ys = [sig[o][key]["power"] for o in OBJECTS]
        with_mean(sx, key, xs, ys, mean_xy=(col, float(np.mean(ys))))
    sx.set_xlim(-0.5, 1.5)
    sx.set_xticks([0, 1])
    sx.set_xticklabels(["Radar\nFields", "RadarSplat§"])
    sx.tick_params(axis="y", labelleft=False)
    sx.set_title("no complex output", fontsize=7, color=MUTED, pad=3)
    grid(sx)
    sx.grid(False, axis="x")


def draw_geometry(ax, numbers, panel):
    """One geometry pair of GEOMETRY_PANELS on `ax`, one marker per (method, object)."""
    geo = numbers["geometry"]
    mx, my, xlabel, ylabel, xlim = GEOMETRY_PANELS[panel]
    for key in GEOMETRY_KEYS[::-1]:
        xs = [geo[o][key][mx] for o in OBJECTS]
        ys = [geo[o][key][my] for o in OBJECTS]
        with_mean(ax, key, xs, ys)
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.02, 1.0 if my == "f1" else 0.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plain_log_ticks(ax.xaxis)
    grid(ax)
    better_notes(ax, "lower", "higher")


def shared_legend(fig, keys, ncol, y_top):
    fig.legend(handles=legend_handles(keys), loc="upper center", bbox_to_anchor=(0.53, y_top), ncol=ncol,
               frameon=False, handletextpad=0.2, columnspacing=0.9)


def signal_figure(numbers, out, formats):
    fig = plt.figure(figsize=(5.5, 3.05))
    ax = fig.add_axes([0.095, 0.14, 0.64, 0.70])
    sx = fig.add_axes([0.765, 0.14, 0.215, 0.70], sharey=ax)
    draw_signal(ax, sx, numbers)
    shared_legend(fig, SIGNAL_COMPLEX + SIGNAL_POWER_ONLY, ncol=6, y_top=1.0)
    save(fig, out / "rift_dataset_signal_scatter", formats)


def geometry_figure(numbers, out, formats):
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.75))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.16, top=0.80, wspace=0.28)
    for ax, panel, tag in zip(axes, GEOMETRY_PANELS, ("(a)", "(b)")):
        draw_geometry(ax, numbers, panel)
        panel_tag(ax, tag)
    rows = [GEOMETRY_KEYS[:4], GEOMETRY_KEYS[4:]]  # legend fills columns first; this makes rows read across
    shared_legend(fig, [k for pair in zip(*rows) for k in pair], ncol=4, y_top=1.0)
    save(fig, out / "rift_dataset_geometry_scatter", formats)


def combined_figure(numbers, out, formats):
    """The paper figure: (a) signal with its strip on top, (b) Chamfer x F1 and (c) HD95 x IoU below.

    Laid out in inches on the 5.5 in text width, so the figure is placed at 100 % and its text keeps its size.
    """
    width, height = 5.5, 4.9
    left, right = 0.52, 5.42

    def axes(x0, y0, w, h, **kw):
        return fig.add_axes([x0 / width, y0 / height, w / width, h / height], **kw)

    fig = plt.figure(figsize=(width, height))
    top_y0, top_h = 2.69, 1.65
    ax = axes(left, top_y0, 3.50, top_h)
    sx = axes(4.20, top_y0, right - 4.20, top_h, sharey=ax)
    draw_signal(ax, sx, numbers)
    panel_tag(ax, "(a)")

    bottom_y0, bottom_h, gap = 0.42, 1.55, 0.60
    w = (right - left - gap) / 2
    for x0, panel, tag in ((left, "chamfer_f1", "(b)"), (left + w + gap, "hd95_iou", "(c)")):
        bx = axes(x0, bottom_y0, w, bottom_h)
        draw_geometry(bx, numbers, panel)
        panel_tag(bx, tag)

    shared_legend(fig, LEGEND_KEYS, ncol=4, y_top=1.0)
    save(fig, out / "rift_dataset_performance_scatter", formats)


# GOTCHA vehicles (Table 9), the same two geometry pairs as Fig. 2 (b)(c), in metres
GOTCHA_VEHICLES = [("camry", "Toyota Camry"), ("sentra", "Nissan Sentra"), ("santafe", "Hyundai Santa Fe")]
GOTCHA_KEYS = ["rift", "spinr", "backprojection"]
GOTCHA_PANELS = {        # x, y, labels, x limits (log), y limits
    "chamfer_f1": ("chamfer", "f1", "Chamfer (m$^2$)", "F1-score", (0.015, 10.0), (-0.025, 1.0)),
    "hd95_iou": ("hd95", "iou", "HD95 (m)", "IoU", (0.1, 10.0), (-0.005, 0.2)),
}


def gotcha_values(camry_json, geometry, records=None):
    """Per-vehicle Table 9 values: every vehicle from the merged Table 9 records (per-vehicle "table"), or the Camry
    from its paper records and the new vehicles from their geometry.json."""
    if records:
        merged = json.loads(Path(records).read_text())
        values = {v: merged[v]["table"] for v, _ in GOTCHA_VEHICLES if v in merged}
    else:
        rows = json.loads(Path(camry_json).read_text())["geometry"]["rows"]
        values = {"camry": {"rift": rows["rift"], "spinr": rows["spinr"], "backprojection": rows["backprojection_full"]}}
    for spec in geometry:
        name, path = spec.split("=", 1)
        values[name] = json.loads(Path(path).read_text())["table"]
    missing = [v for v, _ in GOTCHA_VEHICLES if v not in values]
    if missing:
        raise SystemExit(f"GOTCHA figure: no values for {missing}")
    return values


def gotcha_figure(values, out, formats):
    """GOTCHA geometry across the three vehicles, laid out as the bottom row of the combined figure: faint
    per-vehicle markers, solid three-vehicle means (Table 2's values)."""
    width, height = 5.5, 2.45
    left, right, gap = 0.52, 5.42, 0.60
    w = (right - left - gap) / 2
    fig = plt.figure(figsize=(width, height))
    for i, (panel, tag) in enumerate(zip(GOTCHA_PANELS, ("(a)", "(b)"))):
        mx, my, xlabel, ylabel, xlim, ylim = GOTCHA_PANELS[panel]
        ax = fig.add_axes([(left + i * (w + gap)) / width, 0.42 / height, w / width, 1.55 / height])
        for key in GOTCHA_KEYS[::-1]:
            with_mean(ax, key, [values[v][key][mx] for v, _ in GOTCHA_VEHICLES],
                      [values[v][key][my] for v, _ in GOTCHA_VEHICLES])
        xs = [values[v][k][mx] for v, _ in GOTCHA_VEHICLES for k in GOTCHA_KEYS]
        ys = [values[v][k][my] for v, _ in GOTCHA_VEHICLES for k in GOTCHA_KEYS]
        if not (xlim[0] < min(xs) and max(xs) < xlim[1] and ylim[0] < min(ys) and max(ys) < ylim[1]):
            raise SystemExit(f"GOTCHA figure: a value falls outside the fixed limits of panel {panel}")
        ax.set_xscale("log")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plain_log_ticks(ax.xaxis)
        grid(ax)
        better_notes(ax, "lower", "higher")
        panel_tag(ax, tag)
    shared_legend(fig, GOTCHA_KEYS, ncol=3, y_top=1.0)
    save(fig, out / "gotcha_performance_scatter", formats)


FIGURES = {"combined": combined_figure, "signal": signal_figure, "geometry": geometry_figure,
           "gotcha": None}


def save(fig, stem, formats):
    for ext in formats:
        fig.savefig(f"{stem}.{ext}", dpi=300 if ext == "png" else "figure")
        if ext == "svg":
            retarget_svg_family(f"{stem}.svg")
    plt.close(fig)
    print("wrote", stem, "+".join(formats))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--numbers", help="paper_numbers.json (RIFT-dataset figures)")
    p.add_argument("--gotcha-camry", help="GOTCHA: the Camry paper records (camry_table_numbers.json)")
    p.add_argument("--gotcha-geometry", action="append", default=[],
                   help="GOTCHA: VEHICLE=geometry.json of a new vehicle (sentra, santafe)")
    p.add_argument("--gotcha-records", help="GOTCHA: merged Table 9 records (table9_provisional_records.json), "
                   "used instead of --gotcha-camry/--gotcha-geometry")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--figures", default="combined", help=f"comma list of {','.join(FIGURES)} (default combined)")
    p.add_argument("--formats", default="pdf,svg,png",
                   help="comma list of pdf, svg, png; use pdf,svg for the manuscript tree (default pdf,svg,png)")
    args = p.parse_args()
    figures = args.figures.split(",")
    formats = args.formats.split(",")
    unknown = [f for f in figures if f not in FIGURES] + [f for f in formats if f not in ("pdf", "svg", "png")]
    if unknown:
        p.error(f"unknown figure or format: {unknown}")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    style()
    for name in figures:
        if name == "gotcha":
            gotcha_figure(gotcha_values(args.gotcha_camry, args.gotcha_geometry, args.gotcha_records), out, formats)
        else:
            if not args.numbers:
                p.error(f"--numbers is required for {name}")
            FIGURES[name](json.loads(Path(args.numbers).read_text()), out, formats)


if __name__ == "__main__":
    main()
