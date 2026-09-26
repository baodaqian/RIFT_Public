#!/usr/bin/env python
"""Build both RIFT results decks from one shared source (GT template).

The 2026-07-13 sphere-round deck is the base: it supplies the GT template and
nothing else. Its own 10 slides are STRIPPED (`drop_all_slides`) and the whole
deck is rebuilt programmatically in four parts:

    Part 1  `add_part1_initial_results`  results to date. Sphere / cube / tetra
            compressed to TABLES ONLY (their ~20 render slides are archived in
            the July decks); B787 keeps tables AND visualizations.
    Part 2  `add_competitive_strategy` + `add_literature_review` — the storyline
            from RIFT_RECYCLING_PLAN.md: who the competitors are, what was
            retracted, what "win" means.
    Part 3  `add_part3_round6` — completed Round 6 ablation results.
    Part 4  `add_part4_audited_comparisons` — audited radar/sonar baselines.

Re-run whenever numbers or figures update. Both outputs are regenerated from
scratch, so edit the shared section data here rather than either .pptx:

* ``RIFT_Results_Series.pptx`` — every experiment result retained so far.
* ``RIFT_Current_Best_Results.pptx`` — selected Round-5/6 RIFT B787 models plus
  the complete B787 baseline comparison.

    module load anaconda3 && conda activate RIFT
    python scripts/plot_val_error_curves.py --set round6 \
        --out figures/b787_recon/r6_error_curves.png
    python scripts/render_recon_grid.py --set round6 \
        --out figures/b787_recon/r6_grid.png
    python scripts/render_recon_grid.py --set method_comparison \
        --out figures/b787_recon/method_comparison.png
    python scripts/plot_b787_nvs_sphere.py
    python scripts/build_results_deck.py

Adding a future round = write an `add_partN_*` / `add_roundN_*` function
following `add_part3_round6` and call it in main().
"""
import copy
import json
import os

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

BASE = "slides/RIFT_PEC_Sphere_bw3ghz_Results_2026-07-13.pptx"
OUT = "slides/RIFT_Results_Series.pptx"
OUT_BEST = "slides/RIFT_Current_Best_Results.pptx"
FONT = "Roboto"
SLIDE_W = 13.333


def find_layout(prs, name):
    for m in prs.slide_masters:
        for lay in m.slide_layouts:
            if lay.name == name:
                return lay
    raise KeyError(name)


def add_slide(prs, layout_name, title):
    s = prs.slides.add_slide(find_layout(prs, layout_name))
    body_ph = None
    for ph in list(s.placeholders):
        if ph.placeholder_format.type == 1:  # TITLE
            ph.text_frame.text = title
        else:
            body_ph = ph
    return s, body_ph


def content_slide(prs, title):
    """'Title and Content' slide with the unused body placeholder removed
    (matches how the sphere-round slides were built: manual textboxes only)."""
    s, body = add_slide(prs, "Title and Content", title)
    if body is not None:
        body._element.getparent().remove(body._element)
    return s


def caption(s, text, top, left=0.6, width=12.3, size=13, align_center=False):
    tb = s.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(0.3))
    p = tb.text_frame.paragraphs[0]
    r = p.add_run()
    r.text = text
    r.font.name, r.font.size = FONT, Pt(size)
    if align_center:
        from pptx.enum.text import PP_ALIGN
        p.alignment = PP_ALIGN.CENTER
    return tb


def picture(s, path, top, width=8.4):
    from PIL import Image
    with Image.open(path) as im:
        aspect = im.height / im.width
    left = (SLIDE_W - width) / 2
    return s.shapes.add_picture(path, Inches(left), Inches(top),
                                Inches(width), Inches(width * aspect))


def bullets(s, items, top, left=0.5, width=12.3, height=5.0, size=15):
    """items: list of paragraphs; each paragraph is a list of (text, bold) runs.
    A '•  ' prefix is prepended (sphere-deck convention) unless the first run
    starts with a non-bullet marker '::' (used for plain/heading lines)."""
    height = min(height, 7.35 - top)  # keep the frame on the slide
    tb = s.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, runs in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        plain = runs[0][0].startswith("::")
        for j, (text, bold) in enumerate(runs):
            r = p.add_run()
            if j == 0:
                r.text = text[2:] if plain else "•  " + text
            else:
                r.text = text
            r.font.name, r.font.size, r.font.bold = FONT, Pt(size), bold
    return tb


def table(s, rows, top, left=0.5, width=10.5, header_size=14, body_size=13,
          col_widths=None):
    n_r, n_c = len(rows), len(rows[0])
    height = 0.32 * n_r
    t = s.shapes.add_table(n_r, n_c, Inches(left), Inches(top),
                           Inches(width), Inches(height)).table
    if col_widths:
        for ci, w in enumerate(col_widths):
            t.columns[ci].width = Inches(w)
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = t.cell(ri, ci)
            cell.text = str(val)
            runs = cell.text_frame.paragraphs[0].runs
            if not runs:          # empty cell -- nothing to style
                continue
            r = runs[0]
            r.font.name = FONT
            r.font.size = Pt(header_size if ri == 0 else body_size)
            r.font.bold = ri == 0
    return t


# --------------------------------------------------------------------------
# PART 1 -- results to date, compressed. The simple targets (sphere, cube,
# tetrahedron) are NUMBERS ONLY: their renders lived on ~20 slides and are now
# archived in the July decks. The B787 novel-view work keeps both its tables
# and its visualizations, because it is the live thread.
#
# Provenance of the tables below:
#   SPHERE_SWEEP / SPHERE_INTERP -- 2026-07-13 sphere-round deck (slides 4, 9)
#   CUBE_TETRA_FIT / _MISFIT     -- 2026-07-16 cube/tetra round
#   R5_LADDER / R5_PRUNE         -- scripts/scrape_training_logs.py, 2026-08-06
# --------------------------------------------------------------------------

FIG = "figures"
BR = os.path.join(FIG, "b787_recon")


def load_range_power_metric(label):
    """Read the audited checkpoint rerender rather than duplicating its number.

    Both decks are built together, and both now draw these secondary signal
    scores from the same evaluator artifact.  A missing/incomplete cache stays
    visibly pending instead of silently retaining an old hard-coded value.
    """
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    path = os.path.join(root, "figures", "b787_range_power", f"{label}_metrics.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        result = json.load(handle)
    if int(result.get("views_complete", 0)) != int(result.get("views_total", -1)):
        return None
    return result


RANGE_POWER_RESULTS = {
    "r6": load_range_power_metric("rift_r6_occ_dc"),
    "r5": load_range_power_metric("rift_r5_target20k"),
    "r7": load_range_power_metric("rift_r7_target20k_shdeg1em9"),
    "spinr": load_range_power_metric("spinr_style_deg0"),
}


def range_power_pct(key):
    result = RANGE_POWER_RESULTS[key]
    return (f"{100.0 * result['normalized_range_power_rel_mse']:.4f}%"
            if result is not None else "not computed")


def range_power_reduction(numerator, denominator):
    """Relative error reduction in percent, or ``None`` until both audits finish."""
    first, second = RANGE_POWER_RESULTS[numerator], RANGE_POWER_RESULTS[denominator]
    if first is None or second is None:
        return None
    return 100.0 * (1.0 - first["normalized_range_power_rel_mse"] /
                    second["normalized_range_power_rel_mse"])


def reduction_text(value):
    return f"{value:.2f}%" if value is not None else "pending"


def load_geraf_complex_metric(split):
    """Load a complete direct-response GeRaF audit for one checkpoint split."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    path = os.path.join(
        root,
        "figures",
        "b787_range_power",
        f"geraf_complex_response_{split}_metrics.json",
    )
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        result = json.load(handle)
    expected = 1800 if split == "train" else 200
    if (int(result.get("views_complete", 0)) != expected or
            int(result.get("views_total", -1)) != expected or
            result.get("split") != split or
            result.get("observable") != "raw_complex_frequency_response"):
        return None
    return result


GERAF_COMPLEX_RESULTS = {
    "train": load_geraf_complex_metric("train"),
    "validation": load_geraf_complex_metric("validation"),
}


def geraf_complex_pct(split):
    result = GERAF_COMPLEX_RESULTS[split]
    return (f"{100.0 * result['coherent_complex_rel_mse']:.4f}%"
            if result is not None else "pending")

ACQUISITION = [
    ("target", "scene box", "grid", "train / val views", "fc / bandwidth", "range res"),
    ("PEC sphere, r = 1.0 m", "±1.5 m", "24³ · 48³", "750 / 50", "79 GHz / 3 GHz", "5.0 cm"),
    ("PEC cube, 2 m side", "±2.0 m", "32³ · 64³", "750 / 50", "79 GHz / 3 GHz", "5.0 cm"),
    ("PEC tetrahedron, h = 2 m", "±2.0 m", "32³ · 64³", "750 / 50", "79 GHz / 3 GHz", "5.0 cm"),
    ("B787 airframe, D = 0.10 m", "±0.15 m", "48³", "1800 / 200", "10 GHz / 3 GHz", "5.0 cm"),
]

SPHERE_SWEEP = [
    ("cell", "train rel-MSE ↓", "val rel-MSE ↓", "|g|", "shell peak", "radial FWHM ↓"),
    ("grid · 24³", "94.9%", "105%", "0.81", "1.036 m", "32 cm"),
    ("grid · 48³", "62.9%", "145%", "1.10", "1.036 m", "32 cm"),
    ("grid_sh6 · 24³", "8.1%", "250%", "3.56", "1.036 m", "32 cm"),
    ("grid_sh6 · 48³", "0.54%", "156%", "1.62", "1.036 m", "26 cm"),
    ("point_sh · 24³", "70.3%", "119%", "1.36", "1.036 m", "35 cm"),
    ("point_sh · 48³", "—", "—", "—", "1.036 m", "32 cm"),
]

SPHERE_INTERP = [
    ("scene (grid, held-out views)", "rel-MSE, trained gain ↓", "rel-MSE, optimal gain ↓"),
    ("48³ native — all 50 views", "145.0%  (= training log, exact)", "100.0%"),
    ("48³ native — 10-view subset", "129.6%", "99.7%"),
    ("96³ trilinear — same 10 views", "101.2%", "100.0%"),
]

CUBE_TETRA_FIT = [
    ("cell", "epochs", "train rel-MSE ↓", "val rel-MSE ↓"),
    ("cube · grid · 32³", "150 / 150", "97.8%", "511%"),
    ("cube · grid_sh6 · 32³", "150 / 150", "29.7%", "93,341%"),
    ("cube · grid_sh6 · 32³ · L1 3e-7", "150 / 150", "64.9%", "2,746%"),
]

CUBE_TETRA_MISFIT = [
    ("cell", "E within 5 cm ↑", "E within 10 cm ↑", "conc. vs uniform ↑", "energy peak (signed d)"),
    ("cube grid 32³", "0%*", "56.7%", "6.0×", "+6.3 cm"),
    ("cube grid_sh6 32³", "0%*", "54.9%", "5.8×", "+6.3 cm"),
    ("cube grid_sh6 32³ L1", "0%*", "60.9%", "6.5×", "+6.3 cm"),
    ("cube grid_sh6 64³ (ep 144)", "65.9%", "74.0%", "8.0×", "+3.8 cm"),
    ("tetra grid 32³ (ep 129)", "48.3%", "62.9%", "19.1×", "+1.3 cm"),
    ("tetra grid_sh6 32³ (ep 111)", "51.7%", "64.8%", "19.6×", "+1.3 cm"),
]

R5_LADDER = [
    ("run", "SH degree", "coeffs/voxel", "epochs", "train rel-MSE ↓", "best val rel-MSE ↓"),
    ("R5 · deg 0", "0", "1", "150 / 150  (done)", "5.76%", "28.5%"),
    ("R5 · deg 1", "1", "4", "150 / 150  (done)", "2.32%", "26.0%"),
    ("R5 · deg 2", "2", "9", "150 / 150  (done)", "1.40%", "25.5%  ← best no-prune"),
    ("R4 · deg 3  (physics cap)", "3", "16", "150 / 150  (done)", "0.98%", "26.1%"),
    ("baseline · deg 6 (uncapped)", "6", "49", "150 / 150  (done)", "0.60%", "38.5%"),
]

R5_PRUNE = [
    ("run", "prune mode", "active voxels", "epochs", "train rel-MSE ↓", "best val rel-MSE ↓"),
    ("R5 · target 20k", "target", "110 592 → 20 000", "150 / 150  (done)", "12.5%", "25.3%   ← best overall"),
    ("R5 · target 4k", "target", "110 592 → 4 000", "150 / 150  (done)", "12.1% @ep80", "27.2% @ep80 → 49.7%"),
    ("R5 · mass 0.002", "mass", "110 592 → 4 000", "150 / 150  (done)", "7.6% @ep60", "29.1% @ep60 → 49.7%"),
    ("R4 · deg 6 + prune", "relmax", "104 348", "147 / 150  (stalled)", "1.10%", "35.9%"),
    ("R4 · grow q0.02", "relmax", "→ 235", "150 / 150  (done)", "4.7% @ep29", "26.3% @ep29 → 100%"),
    ("R4 · grow q0.10", "relmax", "→ 68", "150 / 150  (done)", "16.2% @ep39", "26.6% @ep39 → 100%"),
    ("R4 · grow q0.30", "relmax", "→ 71", "150 / 150  (done)", "12.4% @ep39", "26.2% @ep39 → 100%"),
    ("R4 · grow angular", "relmax", "→ 22", "150 / 150  (done)", "9.1% @ep49", "26.4% @ep49 → 100%"),
]


# Scene-reconstruction metrics under the SH-SAS / Reed protocol, computed by
# scripts/eval_b787_geometry_metrics.py (2026-08-06). CD is pytorch3d's
# convention (mean SQUARED NN distance, both directions) as in their tables;
# every metric is reported at its own optimal threshold, which is what Reed's
# `optimal_value_*.csv` files publish. F1 at Reed's FIXED default 0.20 is the
# control column -- the whole point of the next slide.
GEOM_LADDER = [
    ("run", "val rel-MSE ↓", "CD surf ↓", "CD vol ↓", "HD95 mm ↓", "IoU solid ↑", "F1 ↑", "F1 @ 0.20 ↑"),
    ("baseline · deg 6", "38.5%", "7.20e−05", "5.83e−05", "14.2", "0.179", "0.765", "0.257"),
    ("deg 0 (isotropic = SpINR)", "28.5%", "9.69e−05", "9.48e−05", "15.7", "0.111", "0.688", "0.605"),
    ("deg 1", "26.0%", "5.63e−05", "2.79e−05", "13.7", "0.321", "0.836", "0.586"),
    ("deg 2", "25.5%", "5.43e−05", "3.44e−05", "14.2", "0.328", "0.841", "0.504"),
    ("deg 3 (physics cap)", "26.1%", "5.14e−05", "3.25e−05", "10.2", "0.300", "0.843", "0.423"),
    ("deg 3 + prune 20k", "25.3%", "4.86e−05", "3.39e−05", "9.8", "0.296", "0.851", "0.526"),
    ("deg 3 + prune 4k", "27.2%", "5.14e−05", "3.47e−05", "10.0", "0.285", "0.841", "0.449"),
    ("deg 3 + prune mass", "29.1%", "5.14e−05", "3.44e−05", "11.4", "0.301", "0.861", "0.405"),
]

# SH-SAS's published scene-reconstruction numbers, read from arXiv:2509.11087v1
# (2026-08-06). Table 1 = average over their four SIMULATED sonar scenes
# (Buddha, Armadillo, Bunny, XYZ Dragon), reported for both the point-cloud and
# the marching-cubes-mesh variant -- the same two variants our script computes.
# Their Chamfer is printed in scientific notation with NO stated unit, and the
# paper states no mesh scale, scene extent or voxel size anywhere; we checked.
SHSAS_TABLE1 = [
    ("method (dataset)", "variant", "Chamfer ↓", "IoU ↑", "Precision ↑", "F1 ↑"),
    ("Backprojection  (their sim.)", "point cloud", "3.04e−4", "0.201", "0.244", "0.333"),
    ("Reed et al.  (their sim.)", "point cloud", "1.96e−4", "0.447", "0.428", "0.566"),
    ("SH-SAS  (their sim.)", "point cloud", "8.89e−5", "0.497", "0.545", "0.616"),
    ("Backprojection  (their sim.)", "mesh", "3.85e−4", "0.161", "0.225", "0.272"),
    ("Reed et al.  (their sim.)", "mesh", "3.97e−4", "0.133", "0.171", "0.232"),
    ("SH-SAS  (their sim.)", "mesh", "9.70e−5", "0.240", "0.395", "0.384"),
    ("RIFT deg 3 + prune 20k  (B787)", "point cloud", "4.86e−5", "0.296", "0.811", "0.851"),
    ("RIFT deg 6 baseline  (B787)", "point cloud", "7.20e−5", "0.179", "0.706", "0.765"),
    ("RIFT deg 0, isotropic  (B787)", "point cloud", "9.69e−5", "0.111", "0.553", "0.688"),
    ("RIFT deg 3 + prune 20k  (B787)", "mesh", "4.83e−5", "0.332", "0.883", "0.845"),
    ("RIFT deg 6 baseline  (B787)", "mesh", "1.44e−4", "0.192", "0.960", "0.672"),
    ("RIFT deg 0, isotropic  (B787)", "mesh", "8.00e−5", "0.175", "0.614", "0.725"),
]

# Their Table 3, point-cloud rows (arXiv:2509.11087v1). Kept because the SPREAD
# is the argument: their own per-object Chamfer varies 1.9x across four objects
# of the same kind, which is larger than most gaps anyone reports between
# methods -- so a cross-DATASET comparison of the absolute number says nothing.
SHSAS_TABLE3 = [
    ("object", "BP Chamfer ↓", "BP IoU ↑", "BP F1 ↑", "Reed Chamfer ↓", "Reed IoU ↑",
     "Reed F1 ↑", "SH-SAS Chamfer ↓", "SH-SAS IoU ↑", "SH-SAS F1 ↑"),
    ("Armadillo", "8.204e−5", "0.243", "0.391", "7.871e−5", "0.418", "0.544", "7.009e−5", "0.444", "0.614"),
    ("Buddha", "9.546e−4", "0.131", "0.232", "4.832e−4", "0.601", "0.608", "9.238e−5", "0.611", "0.756"),
    ("Bunny", "1.233e−4", "0.227", "0.371", "1.524e−4", "0.412", "0.584", "1.255e−4", "0.398", "0.569"),
    ("XYZ Dragon", "5.679e−5", "0.202", "0.336", "6.834e−5", "0.360", "0.529", "6.752e−5", "0.357", "0.526"),
]

GEOM_R6_20260806_SNAPSHOT = [
    ("arm", "val rel-MSE ↓", "CD surf ↓", "HD95 mm ↓", "IoU solid ↑", "F1 ↑", "F1 @ 0.20 ↑"),
    ("baseline · deg 3, no occlusion", "26.1%", "5.14e−05", "10.2", "0.300", "0.843", "0.423"),
    ("O1 · occ ζ frozen 0.1", "71.2%", "7.42e−03", "164.9", "0.005", "0.060", "0.051"),
    ("O2 · occ ζ frozen 1.0", "100.0%", "3.63e−02", "173.8", "0.000", "0.000", "0.000"),
    ("O3 · occ ζ learned (energy)", "29.0%", "5.03e−05", "10.6", "0.284", "0.859", "0.397"),
    ("O4 · occ ζ learned (DC)", "28.0%", "4.99e−05", "10.1", "0.294", "0.846", "0.402"),
    ("S1 · magnitude λ = 2", "37.1%", "5.21e−05", "10.4", "0.307", "0.853", "0.311"),
    ("S2 · magnitude λ = 2 + warm-up", "41.5%", "5.10e−05", "11.7", "0.311", "0.843", "0.276"),
]


def add_part1_initial_results(prs):
    # -- section title ------------------------------------------------------
    s, sub = add_slide(prs, "Title Slide", "Part 1 — Results through Round 5")
    if sub is not None:
        sub.text_frame.text = ("Daqian Bao — historical results through August 6, 2026 · "
                               "simple geometry, then B787 novel-view synthesis")

    # -- acquisition, all four targets in one table -------------------------
    s = content_slide(prs, "Four targets, one acquisition")
    table(s, ACQUISITION, top=1.45, width=12.2,
          col_widths=(3.1, 1.6, 1.6, 2.2, 2.1, 1.6), body_size=12)
    bullets(s, [
        [("Common to all: 2000-view Fibonacci lattice, 16 Tx × 16 Rx MIMO with exact "
          "per-view element positions, 10 m standoff, 600 frequency samples, "
          "range forward operator in fp64, backprojection init, 150 epochs, ", False),
         ("phase_sign = −1", True)],
        [("Range resolution is c/2B — set by BANDWIDTH, not by sample count. The 3 GHz "
          "regeneration (from 150 MHz) is what turned the sphere from a 1 m blob into a "
          "5 cm-resolution shell; measured point-spread 6.1 cm FWHM (ratio 1.23).", False)],
        [("The simple targets are ", False), ("stepping stones", True),
         (" — sphere for curvature, cube for flat faces, tetra for oblique faces. "
          "The endgame is a SCENE, not one object.", False)],
    ], top=3.55, size=13.5)

    # -- sphere numbers -----------------------------------------------------
    s = content_slide(prs, "PEC sphere — geometry works, generalization does not (6 cells, 150 ep)")
    table(s, SPHERE_SWEEP, top=1.35, width=11.4,
          col_widths=(2.4, 2.0, 1.8, 1.2, 2.0, 2.0), body_size=12)
    bullets(s, [
        [("Shell centred within ", False), ("3.6 cm of the true r = 1.0 m", True),
         (" for EVERY representation; radial FWHM 26–35 cm against a ~9–10 cm budget "
          "(6 cm range PSF, 6.25 cm voxel pitch at 48³) — a ~2× radial smear.", False)],
        [("The +3.5 cm outward bias is ", False), ("init-inherited", True),
         (" — a scene initialised at r = 1.00 stays there. Accepted as a calibration "
          "offset, not an error.", False)],
        [("Every cell's val rel-MSE is ", False), (">100% — worse than predicting zero", True),
         (", and it WORSENS as the training fit improves: grid_sh6 48³ fits train to 0.54% "
          "and reads 156% on held-out views. 49 coefficients/voxel memorize 750 views.", False)],
    ], top=3.85, size=13.5)

    # -- sphere interpolation control ---------------------------------------
    s = content_slide(prs, "Sphere control: densifying by trilinear interpolation does NOT fix validation")
    table(s, SPHERE_INTERP, top=1.55, width=10.6,
          col_widths=(4.2, 3.4, 3.0), body_size=13)
    bullets(s, [
        [("Harness verified against train.py's own val split and metric "
          "(145.0% reproduced to the decimal)", False)],
        [("Optimal-gain rel-MSE ≈ 100% everywhere: the predictions are ", False),
         ("uncorrelated with the held-out views", True),
         (" — the >100% was never a scale or calibration problem.", False)],
        [("The 129.6% → 101.2% “improvement” is ", False), ("shrinkage", True),
         (": interpolation smooths the fast phase structure and attenuates the prediction "
          "toward the predict-zero floor.", False)],
        [("Conclusion that set the whole later programme: the validation gap is ", False),
         ("angular memorization, not spatial discretization", True),
         (" — capacity is the lever, not resolution.", False)],
    ], top=3.30, size=13.5)

    # -- cube + tetra numbers -----------------------------------------------
    s = content_slide(prs, "PEC cube & tetrahedron — the fit, and the same generalization wall")
    table(s, CUBE_TETRA_FIT, top=1.40, width=9.4,
          col_widths=(3.8, 1.8, 2.0, 1.8), body_size=13)
    bullets(s, [
        [("All eight cells reached 150 epochs; per-epoch logs survive for these three only, "
          "so the rest are reported from their checkpoints as geometry (next slide).", False)],
        [("Cube val is ", False), ("flash-dominated", True),
         (": the top-5 views carry 99.8% of total power (3486× the median), so whether they "
          "land in the 50 val views moves the number by orders of magnitude. Tetra top-5 = "
          "84.6% (888×).", False)],
        [("Same wall as the sphere, harder: val ≥ 511% everywhere, and SH-6 memorizes "
          "outright (train 29.7% / val 93,341%). L1 3e-7 helps 34× and is nowhere near "
          "enough.", True)],
    ], top=3.35, size=13.5)

    # -- misfit + bias verdict ----------------------------------------------
    s = content_slide(prs, "…but the energy sits ON the true surface, and flat faces show no fixed bias")
    caption(s, "|w|²-weighted signed distance from exact voxel centres to the true surface "
               "(tetra truth recovered from the data itself, independent of the reconstruction)", 1.24)
    table(s, CUBE_TETRA_MISFIT, top=1.62, width=12.0,
          col_widths=(3.3, 1.9, 2.0, 2.2, 2.6), body_size=12)
    bullets(s, [
        [("*Structurally empty at 32³, not a failure: the nearest voxel centres to an "
          "axis-aligned face sit at exactly ±6.25 cm (half-pitch). Concentration = energy "
          "fraction ÷ volume fraction within 10 cm; 1× would mean no structure.", False)],
        [("Cube energy peak: ", False), ("+6.3 cm at 32³ → +3.8 cm at 64³", True),
         (" — the offset HALVES when the pitch halves, so it is a quantization preference, "
          "not a physics offset. Tetra (oblique faces sample distance continuously): "
          "+1.3 cm ≈ unbiased.", False)],
        [("→ the sphere's outward bias is ", False), ("curvature-specific", True),
         (" (tangent-plane / sagitta), and every PEC reconstruction is hollow — "
          "the physically right answer.", False)],
    ], top=4.15, size=12.5)

    # ===================== B787 =====================
    s = content_slide(prs, "The live target: predict the radar signal at viewpoints never seen")
    bullets(s, [
        [("Headline metric: ", False),
         ("validation rel-MSE on held-out viewpoints", True),
         (" — not geometry of support", False)],
        [("Why this is the claim: Sugavanam & Ertin reconstruct geometry through an SDF but "
          "lose complex amplitude, so they cannot synthesize novel views. We keep the complex "
          "field, so we can.", False)],
        [("Two standing constraints that follow: ", False),
         ("no surface/SDF prior", True),
         (" (matching their geometry without one is the result), and ", False),
         ("trilinear interpolation is for visualization only", True),
         (" — never in the training path", False)],
        [("Data: B787 airframe, largest dimension 0.10 m, 2000-view Fibonacci lattice, "
          "16 Tx × 16 Rx MIMO, 10 GHz / 3 GHz bandwidth, r = 10 m; 1800 train / 200 held-out views", False)],
    ], top=1.6)

    # -- optimization model, three panels -----------------------------------
    s = content_slide(prs, "What we optimize (1/3): the scene model")
    picture(s, os.path.join(BR, "opt_model_scene.png"), top=1.45, width=11.6)

    s = content_slide(prs, "What we optimize (2/3): the signal model")
    picture(s, os.path.join(BR, "opt_model_signal.png"), top=1.45, width=11.6)

    s = content_slide(prs, "What we optimize (3/3): global gain and objective")
    picture(s, os.path.join(BR, "opt_model_objective.png"), top=1.35, width=11.6)

    # -- the error curve ----------------------------------------------------
    s = content_slide(prs, "Round 5: B787 validation error 25.3%, from a 38.5% baseline")
    picture(s, os.path.join(BR, "val_error_curves.png"), top=1.30, width=9.0)
    caption(s, "Every arm differs from the baseline in one variable only; marker = best epoch. "
               "All five runs are now complete at 150 epochs.",
            top=6.52, size=12, align_center=True)

    # -- the degree ladder ---------------------------------------------------
    s = content_slide(prs, "The degree ladder (no pruning) — capacity is the lever")
    table(s, R5_LADDER, top=1.35, width=11.8,
          col_widths=(3.2, 1.3, 1.6, 2.4, 1.7, 1.6), body_size=12)
    bullets(s, [
        [("Capping SH degree at the physics value L ≤ 2k·a is the big effect: "
          "49 coefficients → 16, 5.42M → 1.77M complex parameters, val 38.5% → 26.1%", True)],
        [("The ladder is now complete, and ", False),
         ("degree 2 beats the physics cap of 3", True),
         (" (25.5% vs 26.1%) — the bound is an upper bound, not the optimum. Degree 0 "
          "(isotropic, i.e. the SpINR representation) is the worst rung at 28.5%.", False)],
        [("Note the direction of the train column: the BETTER the training fit, the WORSE "
          "the held-out error. Every row of this table is over-fitting, not under-fitting.", True)],
    ], top=3.6, size=13)

    s = content_slide(prs, "The degree ladder, rendered (dense trilinear interpolation)")
    picture(s, os.path.join(BR, "ladder_grid.png"), top=1.30, width=11.6)
    caption(s, "Degree 0 fragments into speckle; 1–2 give the sharpest fuselage, wings and tail; "
               "6 washes out into a blob. Cyan = STL truth.",
            top=6.55, size=12, align_center=True)

    # -- the prune arms ------------------------------------------------------
    s = content_slide(prs, "The pruning arms — including the ones that collapsed")
    table(s, R5_PRUNE, top=1.30, width=11.8,
          col_widths=(2.4, 1.5, 2.2, 2.2, 1.8, 1.7), body_size=12)
    bullets(s, [
        [("Pruning to 20 000 voxels adds a further 0.8 pt and ", False), ("holds", True),
         (" — but ", False), ("4 000 voxels destroys the fit", True),
         (" (val degrades to 49.7%). The support the coherent sum needs is bracketed "
          "between 4k and 20k; the ~296–4 000 DOF estimate that set those targets is too small.", False)],
        [("The four R4 grow arms all reached ", False), ("26.2–26.6%", True),
         (" — and reached it FAST — before the ", False),
         ("relmax", True), (" ratchet ate them down to 22–235 voxels and the predict-zero floor. "
          "Those scenes were never saved (checkpoint tracked TRAIN loss); every arm since uses "
          "--checkpoint-metric val, so the best scenes survive on disk.", False)],
    ], top=4.35, size=13)

    s = content_slide(prs, "Pruning arms, rendered — 4k voxels is visibly too few")
    picture(s, os.path.join(BR, "prune_grid.png"), top=1.30, width=9.6)
    caption(s, "The 4k and mass panels are their BEST-val epochs (80 and 60), before each "
               "degraded to 49.7%. Degree 6 + prune stalled at epoch 147.",
            top=6.72, size=12, align_center=True)

    # -- dense visualization ------------------------------------------------
    s = content_slide(prs, "Dense visualization: Plenoxel-style trilinear interpolation")
    picture(s, os.path.join(BR, "r5_target20k_dense_overlay.png"), top=1.7, width=12.2)
    caption(s, "Best run (25.3%). The trained 48³ SH grid is trilinearly upsampled 4× to 192³, "
               "then projected as max-intensity energy Σ|c_ℓm|². Cyan = ground-truth STL. "
               "Visualization only — never in the training path.",
            top=5.82, size=12, align_center=True)

    s = content_slide(prs, "Same rendering, baseline vs best: the airframe resolves")
    caption(s, "Top: uncapped SH degree 6, val 38.5% — a diffuse blob.    "
               "Bottom: degree 3 + prune to 20k voxels, val 25.3% — fuselage, wing and tail "
               "cross separate cleanly.",
            top=1.12, size=12, align_center=True)
    picture(s, os.path.join(BR, "baseline_deg6_dense_overlay.png"), top=1.50, width=8.8)
    picture(s, os.path.join(BR, "r5_target20k_dense_overlay.png"), top=4.45, width=8.8)

    # -- SH-SAS / Reed geometry metrics --------------------------------------
    s = content_slide(prs, "Scored on the competitors' own metrics: Chamfer, IoU, F1, Hausdorff")
    table(s, GEOM_LADDER, top=1.28, width=12.4,
          col_widths=(3.0, 1.5, 1.5, 1.5, 1.4, 1.4, 1.0, 1.1),
          header_size=11, body_size=11)
    bullets(s, [
        [("Protocol is ", False), ("theirs, not ours", True),
         (": Reed et al.'s public evaluation code (which SH-SAS's Tables 1/3 inherit) "
          "min-max normalizes the magnitude field, thresholds it, and takes the voxel "
          "centres as a point cloud. Truth = 20 000 surface + 50 000 volume points sampled "
          "from the B787 STL. CD is pytorch3d's mean-SQUARED bidirectional convention.", False)],
        [("Min-max normalization is also what makes this survive our free (gain, scene) "
          "gauge — the metric never sees |w| in absolute units.", False)],
        [("IoU is low for everyone ", False), ("by construction", True),
         (": their GT point cloud is SOLID and a coherent reconstruction is a SHELL. "
          "SpINRv2's own Table 1 reports IoU ≈ 0.09 for every method including "
          "backprojection; we read 0.11–0.33 on the same kind of comparison.", False)],
        [("Absolute values are ", False), ("not portable across papers", True),
         (" — they normalize their meshes and never state the scale, the same defect as "
          "their supplementary Table 4. What transfers is the protocol and the ranking.", False)],
        [("Degree 0 — the isotropic representation SpINR uses — is the worst rung here "
          "(F1 0.688, IoU 0.111) as well as on val. That is the single-variable "
          "angular-model comparison, on identical physics.", True)],
    ], top=4.28, size=11.5)

    # -- head-to-head against SH-SAS's published numbers ---------------------
    s = content_slide(prs, "Side by side with SH-SAS's published scene numbers")
    caption(s, "↓ lower is better  ·  ↑ higher is better.   Theirs: arXiv 2509.11087 Table 1, "
               "averaged over four SIMULATED sonar scenes. Ours: B787, same protocol, each metric "
               "at its own optimal threshold; our P/F1 tolerance is 6.25 mm = one voxel pitch = "
               "6.25% of the target's largest dimension, theirs is not stated.", 1.14, size=11.5)
    table(s, SHSAS_TABLE1, top=1.50, width=12.0,
          col_widths=(4.0, 2.0, 1.6, 1.4, 1.6, 1.4), header_size=11, body_size=10.5)
    bullets(s, [
        [("This is NOT a head-to-head and must not be shown as one.", True),
         (" Different target, different modality, different aperture — and, checked in the "
          "paper this week, ", False),
         ("they state no mesh scale, scene extent or voxel size anywhere", True),
         (". Chamfer carries units of length², so without a scale the two columns cannot be "
          "converted in either direction.", False)],
        [("Read the ", False), ("directions", True),
         (", not the gaps: we sit lower on Chamfer and higher on F1 but LOWER on IoU. "
          "Winning two of three and losing the third against the same reference is the "
          "signature of a dataset difference, not a method difference.", False)],
    ], top=5.78, size=11.5)

    # -- their per-object spread ---------------------------------------------
    s = content_slide(prs, "Why the absolute numbers cannot travel: their own per-object spread")
    caption(s, "SH-SAS Table 3, point-cloud rows (arXiv 2509.11087).   ↓ lower is better  ·  "
               "↑ higher is better.", 1.20, size=12)
    table(s, SHSAS_TABLE3, top=1.58, width=12.4,
          col_widths=(1.6, 1.4, 1.0, 1.0, 1.5, 1.1, 1.0, 1.6, 1.2, 1.2),
          header_size=9.5, body_size=10.5)
    bullets(s, [
        [("Across four objects of the same kind, in the same pipeline, ", False),
         ("their own Chamfer moves 1.9×", True),
         (" (7.0e−5 on Armadillo to 1.26e−4 on Bunny), IoU 0.357→0.611 and F1 0.526→0.756. "
          "That within-paper spread is larger than the gap they report over Reed on three "
          "of the four objects.", False)],
        [("On Bunny and Dragon, ", False),
         ("classical backprojection beats them on Chamfer", True),
         (" (1.233e−4 vs 1.255e−4; 5.679e−5 vs 6.752e−5) — the same pattern SpINRv2 shows, "
          "where backprojection wins Hausdorff outright. Their own text concedes they rank "
          "best “on most metrics and is never worst.”", False)],
        [("Conclusion for our paper: quote these as ", False), ("context, never as a target", True),
         (". A real comparison needs ", False), ("both methods on the same data", True),
         (" — Gate S1's AirSAS route, which is public through Reed's repo — or a margin over "
          "BACKPROJECTION, which both papers carry as a row and which is a ratio rather than "
          "an absolute.", False)],
    ], top=4.10, size=11.5)

    # -- geometry vs signal --------------------------------------------------
    s = content_slide(prs, "…and the metrics do NOT rank what held-out signal error ranks")
    picture(s, os.path.join(BR, "geometry_vs_val.png"), top=1.28, width=9.8)
    bullets(s, [
        [("Report each arm at ", False), ("its own best threshold", True),
         (", as they do, and geometry is uncorrelated with novel-view signal error "
          "(ρ = +0.12, p = 0.71 — and the WRONG sign). Hold the threshold ", False),
         ("fixed", True), (" and it correlates strongly (ρ = −0.79, p = 0.002).", False)],
        [("Mechanism: per-arm threshold optimization keeps only each field's best level "
          "set and throws away how much energy sits OFF the target — which is exactly the "
          "part that predicts signal fidelity. The draft's own §3.4 line, measured: "
          "signal error and scene error are not the same axis.", True)],
    ], top=5.72, size=11.5)

    # -- where Part 1 leaves us ---------------------------------------------
    s = content_slide(prs, "Where Round 5 left us — before the final audit")
    bullets(s, [
        [("::Established", True)],
        [("Geometry is solved to the resolution budget on every target: hollow, "
          "surface-concentrated reconstructions on curved, flat and oblique faces, "
          "with no surface prior of any kind", False)],
        [("Held-out-view rel-MSE 38.5% → ", False), ("25.3%", True),
         (" on the B787; the reconstruction is visibly an aircraft, not a blob", False)],
        [("The model was over-parameterized by 10³–10⁴×. The physics cap L ≤ 2k·a is the "
          "fix, and degree 2 beats even the cap value 3", False)],
        [("Pruning needs a fixed point: ", False), ("target", True),
         (" mode converges and holds; ", False), ("relmax", True),
         (" is a ratchet that destroyed four Round-4 arms down to 22 voxels", False)],
        [("Scored on the competitors' own geometry protocol (Chamfer / IoU / F1 / "
          "Hausdorff), the isotropic degree-0 representation — SpINR's — is the worst "
          "rung, on identical physics", False)],
        [("::Open", True)],
        [("Those geometry metrics, reported the way Reed and SH-SAS report them, are "
          "UNCORRELATED with held-out-view signal error (ρ = +0.12, p = 0.71). Beating "
          "them on Chamfer and beating them on novel-view synthesis are two different "
          "results — and we should say which one we are claiming", True)],
        [("Support size sits between 4 000 (breaks) and 20 000 (works) voxels — "
          "the useful bracket is untested", False)],
        [("Train/val gap is still ~2× at best (12.5% vs 25.3%), and the good fits keep most "
          "of their energy OFF the airframe and need it — a coherent sum requires its "
          "small terms", False)],
        [("Every dataset here is ", False), ("PEC and simulated", True),
         (". That is the threat Part 2 has to answer.", False)],
    ], top=1.5, size=14)


# --------------------------------------------------------------------------
# Literature review of the two vision-augmented directions (2026-08-05).
# Text-only section; sources are listed on the last slide of the section.
# --------------------------------------------------------------------------

LIT_MAP = [
    ("work / venue", "modality", "rendered quantity", "geometry prior", "novel-view eval", "our plan"),
    ("SpINR, arXiv 2503.23313", "FMCW radar, turntable", "COMPLEX beat signal", "none (isotropic σ)", "NONE — no NVS at all", "PORT — closest work"),
    ("SH-SAS, arXiv 2509.11087 (Sep 2025)", "synth. ap. SONAR", "COMPLEX transient", "normals + Lambert + TV", "YES — Re/Im, suppl. T4", "PORT + run on AirSAS"),
    ("Reed et al., ACM TOG 42(4) 2023", "synth. ap. SONAR", "COMPLEX, isotropic", "none", "no", "ANCHOR — code public"),
    ("Sugavanam & Ertin, arXiv 2602.17556", "X-band SAR", "phase history, via SDF", "SDF + smoothness", "NO — qualitative only", "PORT — SDF control"),
    ("SAR-GS 2506.21633 · ISPRS 3DGS-SAR 2026", "SAR", "amplitude image only", "3D Gaussians + SH", "PSNR / SSIM / LPIPS", "PORT — ⚠ Gaussians"),
    ("Radar Fields, SIGGRAPH 2024", "FMCW scanning, static", "real power |FFT|", "none", "RMSE/PSNR + CD/RCD", "PORT — power-only rep."),
    ("DART, CVPR 2024", "mmWave FMCW", "range-Doppler magnitude", "none", "yes (magnitude)", "DROP — Doppler"),
    ("RF4D, arXiv 2505.20967", "FMCW, DYNAMIC", "real power (dB)", "none", "PSNR/RMSE + CD/RCD", "DROP — dynamic scene"),
    ("RadarSim, arXiv 2605.26328", "single-chip mmWave", "range-Doppler-azimuth", "CAMERA field + BRDF", "SSIM / PSNR", "DROP — Doppler + camera"),
    ("GeRaF 2.0 “Seeing through boxes”, CVPR 2026", "RF, NLoS", "power heatmaps", "VISUAL LoS prior + SDF", "yes — POWER only", "DROP — NLoS scope"),
    ("RFconstruct, BMVC 2025", "2× COTS auto radar", "point cloud → mesh", "per-class enc–dec", "no signal synthesis", "DROP — needs corpus"),
    ("Neural Surf. & Refl., IROS 2026", "automotive radar", "intensity only", "SDF", "NO — known poses only", "DROP — sparse pt cloud"),
    ("RIFT (this work)", "mmWave near-field MIMO", "COMPLEX S(f)[rx,tx]", "NONE", "held-out-view rel-MSE", "—"),
]

LIT_SOURCES = [
    "Sugavanam & Ertin, “Neural Implicit Representations for 3D SAR Imaging”, arXiv:2602.17556",
    "SH-SAS, “Implicit Neural Representation for Complex Spherical-Harmonic Scattering Fields for 3D SAS”, arXiv:2509.11087",
    "Borts et al., “Radar Fields: Frequency-Space Neural Scene Representations for FMCW Radar”, SIGGRAPH 2024 (arXiv:2405.04662)",
    "Huang et al., “DART: Implicit Doppler Tomography for Radar Novel View Synthesis”, CVPR 2024 (arXiv:2403.03896)",
    "“RF4D: Neural Radar Fields for NVS in Outdoor Dynamic Scenes”, arXiv:2505.20967",
    "“RadarSim: Simulating Single-Chip Radar via Multimodal Neural Fields”, arXiv:2605.26328",
    "“SAR-GS: 3D Gaussian Splatting for SAR Target Reconstruction”, arXiv:2506.21633; ISPRS J. P&RS 231:167 (2026)",
    "“Seeing through boxes: NLoS 3D Reconstruction from Radar Signals” (GeRaF 2.0), CVPR 2026 (arXiv:2605.29098, 2605.29097)",
    "Hussein, Guan, Narashiman, Gupta, Al Hassanieh, “3D Shape Reconstruction from Autonomous Driving Radars” (RFconstruct), BMVC 2025 (arXiv:2504.12348)",
    "“Neural Surface and Reflectance Modelling from 3D Radar Data”, IROS 2026 (arXiv:2603.25623)",
    "RadarSFD arXiv:2509.18068 · RaLD arXiv:2511.07067 · mmDEAR arXiv:2503.02375 · DREAM-PCD arXiv:2309.15374 · RadarGen arXiv:2512.17897",
    "GeoDiff-SAR arXiv:2601.03499 · GeoDiff-SAR II arXiv:2605.21116 · ASC-evolution SAR view completion (Feb 2026)",
    "Bucci & Franceschetti, “On the spatial bandwidth of scattered fields”, IEEE TAP 1987; “On the degrees of freedom of scattered fields”, IEEE TAP 1989",
    "Potter & Moses, “Attributed scattering centers for SAR ATR”, IEEE TIP 1997",
    "Reed, Kim, Blanford, Pediredla, Brown, Jayasuriya, “Neural Volumetric Reconstruction for "
    "Coherent Synthetic Aperture Sonar”, ACM TOG 42(4) 2023 (arXiv:2306.09909) — CODE PUBLIC: "
    "github.com/awreed/Neural-Volumetric-Reconstruction-for-Coherent-SAS",
    "Blanford et al., “An in-air synthetic aperture sonar dataset of target scattering in "
    "environments of varying complexity”, Scientific Data 11:1196 (2024) — AirSAS, PUBLIC: "
    "figshare 10.6084/m9.figshare.26961892 · github.com/tblanford/airsas",
    "Takawale & Roy, “SpINR: Neural Volumetric Reconstruction for FMCW Radars”, arXiv:2503.23313 "
    "— uncatalogued direct competitor, READ OWED",
]


def add_literature_review(prs):
    # -- section title ------------------------------------------------------
    s, sub = add_slide(prs, "Title Slide",
                       "Part 2, continued — Where This Sits in the Literature")
    if sub is not None:
        sub.text_frame.text = ("Daqian Bao — August 6, 2026 · what is open, what is SOTA, "
                               "and which vision direction is worth taking")

    # -- the map ------------------------------------------------------------
    s = content_slide(prs, "The full audit — every work read from the primary source, and what we do about it")
    table(s, LIT_MAP, top=1.10, width=12.4,
          col_widths=(3.1, 1.7, 2.0, 1.8, 1.9, 1.9), header_size=9.5, body_size=8.5)
    bullets(s, [
        [("::THE HEADLINE:  nobody in RADAR does coherent complex-signal novel-view synthesis.", True)],
        [("SpINR reconstructs coherently and then never synthesizes novel views; GeRaF holds out "
          "views but synthesizes POWER heatmaps; Sugavanam & Ertin say they structurally cannot; "
          "the rest discard phase. So the claim is ", False),
         ("“first coherent complex-signal NVS in radar”", True),
         (" — NOT “first complex SH field”, which SH-SAS holds in sonar.", False)],
        [("::DROPPED — Doppler / dynamic scenes, out of scope for now", True)],
        [("DART", True), (" — Doppler tomography is constitutive; our static MIMO array has no "
          "Doppler axis.  ", False), ("RF4D", True),
         (" — dynamic scene with a scene-flow module.  ", False), ("RadarSim", True),
         (" — range-Doppler-azimuth cubes, and needs a paired camera field.", False)],
        [("::DROPPED — structural, not scope", True)],
        [("GeRaF 2.0", True), (" needs an occluding box (NLoS) + a vision prior.  ", False),
         ("RFconstruct", True), (" is a supervised class prior needing a ShapeNet-scale corpus — "
          "we have ~5 targets.  ", False), ("IROS 2603.25623", True),
         (" works from sparse CFAR point clouds and does no NVS.", False)],
        [("::PORTED — we reimplement the method and run it on OUR data", True)],
        [("Missing source code is not a blocker. Our data is the common ground precisely because "
          "we hold ", False), ("exact ground truth", True),
         (" (analytic sphere/cube/tetra, B787 STL) — stronger than the LiDAR and Scaniverse truth "
          "most of them score against.", False)],
    ], top=5.65, size=10.5)

    # -- the collision ------------------------------------------------------
    s = content_slide(prs, "The one real collision: SH-SAS (Sep 2025) — read this first")
    bullets(s, [
        [("SH-SAS is ", False), ("the same representation we built", True),
         (", one modality over: complex spherical-harmonic scattering field, coherent "
          "novel-view synthesis of the complex signal, evaluated on Re/Im directly.", False)],
        [("::Differences that matter", True)],
        [("Sonar, not radar. Multi-resolution hash grid + MLP, not an explicit voxel/point grid.", False)],
        [("They pick ", False), ("SH degree L = 3 by ablation over {1,2,3}", True),
         (" — their paper contains ", False),
         ("no bandwidth argument and no cap criterion", True), (".", False)],
        [("Their own future work: “testing on ", False), ("radar", True),
         (" by swapping modality-specific forward models.”", False)],
        [("::Consequences for us", True)],
        [("We can NO LONGER claim “first complex-valued SH scattering field.” "
          "It must be cited, and the framing has to move off the representation.", True)],
        [("But their L = 3 ablation is our result arriving by accident — and we have the "
          "reason WHY, plus the measurement, plus the fact that finer grids need LOWER degree.", True)],
        [("::READ THE SUPPLEMENT — the main body is misleading on its own (2026-08-05 full read)", True)],
        [("Main body evaluates ", False), ("geometry only", True),
         (" (Chamfer / IoU / Precision / F1). The SH ablation and the ", False),
         ("novel-view signal evaluation (§13, Table 4)", True),
         (" are in the supplementary material.", False)],
    ], top=1.45, size=13.5)

    # -- the standalone SP paper --------------------------------------------
    s = content_slide(prs, "Q1 — The standalone signal-processing paper, from what is already on disk")
    bullets(s, [
        [("Spine: ", False), ("how many angular degrees of freedom does a radar voxel have?", True),
         ("  L ≤ 2k_max·a is the space-bandwidth product of one voxel — the "
          "Bucci–Franceschetti NDF result (TAP 1987/1989) and the Mie N ≈ kR "
          "truncation rule, transplanted to a neural scene representation.", False)],
        [("The measurement that carries it: 49 → 16 coefficients, 5.42M → 1.77M complex "
          "params, val ", False), ("38.5% → 26.1%", True),
         (" — and degree 2 beats the cap value 3, so the bound is an upper bound, "
          "not the optimum.", False)],
        [("::Three supporting results nobody has published", True)],
        [("Model order: the coherent support is bracketed — 4 000 voxels BREAKS the fit, "
          "20 000 works. And the good fit keeps ", False), ("62% of its energy off the airframe "
          "and needs it", True), (": a coherent sum requires its small terms. "
          "That reframes “artifacts” as basis structure.", False)],
        [("Angular sampling law: the n = 100/200/400/800/1800 transition — the plenoptic-"
          "sampling result for COHERENT radar. Radar Fields fits one trajectory; SH-SAS "
          "never varies view count. ", False), ("No radar-NVS paper reports a view-density curve.", True)],
        [("Two negative results with teeth: the (gain, scene) gauge is an exactly flat direction "
          "and renormalizing it silently divides the scene lr by 318× (Adam is invariant to "
          "rescaling the GRADIENT, not the PARAMETER); and relmax pruning is a ratchet with no "
          "fixed point whose damage is ", False), ("invisible in the energy metric", True), (".", False)],
        [("Venue fit: IEEE Trans. Computational Imaging (best), or TAES / ICASSP / Radar Conf / "
          "Asilomar for the short version. ", False),
         ("This paper needs no vision data and no new capability.", True)],
    ], top=1.35, size=13)

    # -- direction 1 --------------------------------------------------------
    s = content_slide(prs, "Q2a — Direction 1: vision-densified scatterers + coherent NVS")
    bullets(s, [
        [("::Who is already there", True)],
        [("RadarSim (May 2026) is the closest: camera neural field initializes a radar "
          "reflectance field, BRDF + learned normals, beats DART and Radar Fields. But it "
          "outputs ", False), ("processed range-Doppler cubes, not coherent signal", True),
         (", and the BRDF+normals model IS a surface prior.", False)],
        [("GeRaF 2.0 / “Seeing through boxes” (CVPR 2026) injects ", False),
         ("visual line-of-sight priors", True),
         (" into a neural RF SDF — geometry, not signal. Radar Fields’ own future work "
          "asks for exactly this cross-modal supervision.", False)],
        [("RadarGen / SDCM / Rad-GS / RPGFusion: camera → radar point cloud. All incoherent, "
          "all detection-driven.", False)],
        [("::What is genuinely open — and it is a physics question, not an engineering one", True)],
        [("Vision gives the ", False), ("optical surface", True),
         (". Radar returns come from specular flashes, edge diffraction, dihedral/trihedral "
          "multipath and creeping waves — ", False),
         ("which are not co-located with the visible surface", True),
         (". Our own 62%-off-airframe measurement says so directly.", False)],
        [("So the interesting form of Direction 1 is not “add detail” — it is ", False),
         ("learn the offset field between the optical surface and the radar phase centers", True),
         (". That is the classical scattering-center localization question (Potter & Moses) "
          "posed as a learned field, and it is open in both communities.", False)],
        [("::Two things to be honest about", True)],
        [("It dilutes the “from the radar signal alone” claim. Keep radar-only as the "
          "headline and let vision enter as DATA, not as a smoothness prior.", False)],
        [("We have ", False), ("no paired vision+radar data", True),
         (" — but we can render paired RGB/depth from data/B787.stl for free, which makes "
          "this a fully controllable oracle study.", False)],
    ], top=1.30, size=12.5)

    # -- direction 2 --------------------------------------------------------
    s = content_slide(prs, "Q2b — Direction 2: class shape prior + point completion (RFconstruct-style)")
    bullets(s, [
        [("::This one is largely closed as described, and it is the weaker direction", True)],
        [("RFconstruct (BMVC 2025) already does radar → 3D shape with a per-category "
          "encoder–decoder for cars, bikes, motorcycles, pedestrians, validated against "
          "depth-camera and LiDAR meshes.", False)],
        [("Point-completion-with-shape-priors is mature and crowded (PCN → SPAC-Net → "
          "3DMambaComplete), and 2025–26 has already moved past it to ", False),
         ("latent-diffusion priors", True),
         (": RadarSFD (SOTA, pretrained depth priors), RaLD, mmDEAR, DREAM-PCD.", False)],
        [("On the SAR side the analogue is done too: ASC-prior view completion for unseen aspect "
          "angles (Feb 2026), GeoDiff-SAR I/II.", False)],
        [("::Why it breaks our differentiator", True)],
        [("A category prior means the answer comes from the prior, not from the signal. "
          "We could no longer claim novel-view synthesis ", False),
         ("from the signal model", True),
         (" — the first reviewer request will be to ablate the prior away, and that lands "
          "us back in Direction 1.", False)],
        [("It also cannot be scored with our headline metric: Chamfer distance to a mesh, "
          "not held-out-view rel-MSE.", False)],
        [("::The only version worth defending", True)],
        [("A generative prior over the ", False), ("SH coefficient field", True),
         (" — not over shape — used as a regularizer in the inverse problem and judged "
          "purely by held-out-view rel-MSE. That is a plug-and-play / score-based-prior paper "
          "(cf. ScoreField, neural inverse scattering with score priors), squarely signal "
          "processing, and undone for coherent view-dependent scattering fields. Bigger lift: "
          "it needs a corpus of scenes.", False)],
    ], top=1.35, size=12.5)

    # -- recommendation -----------------------------------------------------
    s = content_slide(prs, "Q3 — Recommendation")
    bullets(s, [
        [("::1. Write the signal-processing paper now — it is already earned", True)],
        [("The SH-degree bound is the spine; support bracketing, the sampling law and the two "
          "negative results are the body. Cheapest missing number: finish the stalled deg 0/1/2 "
          "ladder. Nothing here needs vision.", False)],
        [("::2. Then Direction 1, but restricted to the oracle-geometry ablation", True)],
        [("Render paired optical views from B787.stl; constrain/initialize support from the "
          "visible surface; measure (a) does held-out rel-MSE improve, and (b) ", False),
         ("where do the radar phase centers sit relative to the optical surface?", True)],
        [("(b) is a result whichever way (a) goes — and it is the paper the vision community "
          "would actually find interesting, because it says what radar sees that a camera "
          "cannot place.", False)],
        [("::3. Drop Direction 2 as framed", True)],
        [("Occupied by RFconstruct and the diffusion-prior line, and it costs us the "
          "“from the signal” claim. Revisit only as a learned prior on the SH field.", False)],
        [("::Standing constraints, re-checked against the literature — both survive", True)],
        [("No surface/SDF prior: still the right call. It is exactly what separates us from "
          "Sugavanam & Ertin, GeRaF 2.0 and RadarSim, all three of which buy geometry with a "
          "surface model and lose the coherent field.", False)],
        [("Trilinear interpolation for visualization only: unchanged.", False)],
    ], top=1.45, size=13.5)

    # -- sources ------------------------------------------------------------
    s = content_slide(prs, "Sources (literature search, 2026-08-05)")
    bullets(s, [[("::" + src, False)] for src in LIT_SOURCES], top=1.30, size=11.5)


# --------------------------------------------------------------------------
# Competitive strategy + the plan (2026-08-05). Written for the advisor
# discussion: who the competitors actually are, what we retracted, what
# survives, and what "win" means. Mirrors RIFT_RECYCLING_PLAN.md §4 and §5.
# --------------------------------------------------------------------------

SHSAS_ANATOMY = [
    ("component", "SH-SAS (arXiv 2509.11087)", "RIFT (this work)"),
    ("representation", "hash grid, 16 lvl → 4096, + MLP 2×32", "explicit voxel grid / point list"),
    ("angular model", "complex SH, L = 3 FIXED (ablation {1,2,3})", "complex SH, L ≤ 2k·a DERIVED"),
    ("supervision", "raw 1-D time-of-flight signal", "raw complex S(f)[rx,tx]"),
    ("propagation", "scalar Helmholtz, ellipsoidal ToF", "scalar Helmholtz, exact near-field MIMO"),
    ("occlusion", "YES — T = Π exp(−ρ̂·Δl), their Eq. 7", "NONE — this is OUR gap"),
    ("opacity", "ρ̂ = |σ_DC|·ζ   (DC-keyed, hand-set ζ)", "— (energy-keyed Σ|c_lm|² proposed)"),
    ("directivity", "Lambertian; normals from ∇|σ_DC|", "free SH lobe, no surface model"),
    ("priors", "L1 sparsity + TV on density, amp, PHASE", "optional group-L1; no TV, no surface"),
    ("evaluation", "Chamfer/IoU/Prec/F1 + Re/Im (suppl. T4)", "held-out-view rel-MSE"),
    ("regime", "sonar 20 kHz, diffuse 3-D prints, REAL", "mmWave 79 GHz, PEC, simulated"),
]

SHSAS_TABLE4 = [
    ("Method", "Render ms ↓", "L1 real ↓", "L1 imag ↓", "L1 abs ↓", "MSE real ↓", "MSE imag ↓", "MSE abs ↓"),
    ("Reed et al. 2023", "147.40", "0.417", "0.639", "0.865", "1.339", "1.346", "2.476"),
    ("SH-SAS", "126.11", "0.402", "0.576", "0.665", "0.797", "0.810", "1.171"),
]


def add_competitive_strategy(prs):
    # -- section title ------------------------------------------------------
    s, sub = add_slide(prs, "Title Slide", "Part 2 — The Storyline: Achieve SOTA on Two Datasets")
    if sub is not None:
        sub.text_frame.text = ("Daqian Bao — August 6, 2026 · who the competitors are, what we "
                               "retracted, what survives, and what “win” means")

    # -- the storyline ------------------------------------------------------
    s = content_slide(prs, "The storyline, in one slide")
    bullets(s, [
        [("::1.  The problem is real and the field is nearly empty — but not empty.", True)],
        [("Coherent novel-view synthesis (predict the COMPLEX field at an unmeasured "
          "viewpoint) is occupied by exactly one method: SH-SAS, in sonar. Everyone else "
          "discards phase before the loss.", False)],
        [("::2.  We are NOT novel against SH-SAS — on the model or on the task.", True)],
        [("Our forward operator is a scalar-Helmholtz Born kernel — the same physics they use. "
          "Both of us train on the raw complex signal. Both of us interpolate held-out views "
          "inside a sampled aperture.", False)],
        [("::3.  Therefore the paper must be a WIN, not a claim of novelty.", True)],
        [("“We don’t have to be totally novel. But if we are not totally novel, then we need to "
          "outperform. It also forms a paper.”", False)],
        [("::4.  The bar: beat SH-SAS on OUR radar data AND on AirSAS (public, real, diffuse).", True)],
        [("One dataset is arguable. Both brackets the specular↔diffuse continuum and kills the "
          "“you only win on PEC” objection in the same move.", False)],
        [("::5.  What we bring that they don’t: the order is DERIVED, not searched.", True)],
        [("L ≤ 2k·a is a per-voxel space-bandwidth product (Bucci–Franceschetti NDF). "
          "It predicts their L = 3 from their bandwidth — on their own data.", True)],
    ], top=1.25, size=12.5)

    # -- anatomy of the competitor -------------------------------------------
    s = content_slide(prs, "Anatomy of the competitor: SH-SAS, component by component")
    table(s, SHSAS_ANATOMY, top=1.22, width=12.4,
          col_widths=(2.1, 5.2, 5.1), header_size=12, body_size=10.5)
    bullets(s, [
        [("Read from the paper body AND the supplement. The main body evaluates ", False),
         ("geometry only", True), ("; the SH ablation and the novel-view signal evaluation "
          "are in the supplementary material.", False)],
    ], top=5.30, size=12)

    # -- the number to beat --------------------------------------------------
    s = content_slide(prs, "The number to beat — SH-SAS supplementary Table 4 (Armadillo)")
    caption(s, "↓ lower is better on every column.", 1.10, size=12)
    table(s, SHSAS_TABLE4, top=1.30, width=12.2,
          col_widths=(2.4, 1.5, 1.4, 1.4, 1.4, 1.4, 1.4, 1.3),
          header_size=12, body_size=11.5)
    bullets(s, [
        [("::Three reasons this is a SOFT but ILL-DEFINED target", True)],
        [("The numbers are ", False), ("UNNORMALIZED", True),
         (" — no signal scale is given anywhere. They cannot be converted to a relative error, "
          "and our 25.3% cannot be converted into their units, without reproducing their exact "
          "preprocessing. Recovering the normalization is a prerequisite, not a detail.", False)],
        [("::", False)],
        [("One object, one baseline. Armadillo only, vs Reed et al. only — no backprojection "
          "row, even though backprojection beats them on Chamfer on 2 of 4 objects in Table 3, "
          "and their own text concedes they rank best “on MOST metrics and is never worst.”", False)],
        [("Their Fig. 12 shows visible ", False), ("amplitude under-prediction", True),
         (" for BOTH methods — predicted traces sit below the ground-truth peaks. The bar looks "
          "low. Defining the normalized metric is itself available to us as a contribution.", False)],
    ], top=2.65, size=12)

    # -- their stated limitations --------------------------------------------
    s = content_slide(prs, "SH-SAS’s own stated limitations — quoted, and they are our openings")
    bullets(s, [
        [("::On noise — this is our strongest attack surface", True)],
        [("“Our approach assumes ", False), ("moderate SNR", True),
         (" and sufficient bandwidth. In very noisy acquisitions, the ", False),
         ("higher-order SH terms (ℓ>0) act like high-frequency angular basis functions that "
          "readily fit noise", True),
         (", introducing spurious anisotropy and ripples that corrupt the geometry.”", False)],
        [("That failure mode is exactly what a bandwidth-derived cap removes ", False),
         ("by construction rather than by tuning", True),
         (". Their L = 3 is picked by ablation, so it has no principled behaviour as SNR falls.", False)],
        [("::On bandwidth", True)],
        [("“With narrowband signals the problem worsens: range resolution degrades as "
          "ΔR ≈ c/(2Δf) … the directional components become weakly identifiable, making the "
          "SH fit ", False), ("ill-conditioned", True), (".”", False)],
        [("::On physics", True)],
        [("They concede they “do not model complex acoustic wave phenomena (e.g. ", False),
         ("diffraction, interference, and multiple reflections/scattering", True), (").”", False)],
        [("::On their own SH order", True)],
        [("Supplement §11: L ∈ {1,2,3}, “as SH level increases, reconstruction quality improves, "
          "with level 3 performing best in our setup.” ", False),
         ("An ablation that runs out at 3 — not a bound.", True)],
    ], top=1.28, size=12)

    # -- the prediction ------------------------------------------------------
    s = content_slide(prs, "The single best argument we have: the bound PREDICTS their L = 3")
    bullets(s, [
        [("::The arithmetic, fixed before the test", True)],
        [("AirSAS transmits an LFM sweep of ", False), ("10 – 30 kHz in air", True),
         (", so λ_min = c / f_max ≈ 343 / 30 000 ≈ ", False), ("11.4 mm", True),
         (", giving k_max = 2π/λ_min ≈ ", False), ("550 rad/m", True), (".", False)],
        [("Our bound L ≤ 2·k_max·a then puts ", False), ("L = 3 at a ≈ 2.7 mm", True),
         (", i.e. a voxel of ≈ 5.5 mm — about 37 voxels across their 0.2 m target. "
          "An entirely natural discretization.", False)],
        [("::Why this matters more than another point on our own sweep", True)],
        [("They found L = 3 ", False), ("empirically, by ablation, with no bandwidth argument", True),
         (". Our bound derives it ", False), ("from their bandwidth alone, on their own data", True),
         (".", False)],
        [("A reviewer cannot attribute that to our tuning — it explains a ", False),
         ("competitor’s published hyperparameter", True), (".", False)],
        [("It is a ", False), ("genuine falsification risk", True),
         (": the arithmetic is fixed in advance, so the test can fail.", False)],
        [("::The caveat to state honestly", True)],
        [("A hash grid has no well-defined voxel scale ‘a’ — which is precisely our "
          "model-order argument (S3). The test is clean only in an explicit grid.", False)],
    ], top=1.30, size=12.5)

    # -- retractions ---------------------------------------------------------
    s = content_slide(prs, "Three claims we retracted this week — say these before a reviewer does")
    bullets(s, [
        [("::1.  “Our forward model is a contribution.”  ", True), ("FALSE.", True)],
        [("It is a scalar-Helmholtz Born kernel — the same physics SH-SAS uses, and MORE exactly "
          "right for sonar than for radar (EM obeys the VECTOR Helmholtz equation; our scalar "
          "kernel silently drops polarization). Porting to sonar removes an approximation.", False)],
        [("::2.  “We extrapolate to viewpoints outside the training aperture.”  ", True),
         ("FALSE.", True)],
        [("Fibonacci sampling covers the whole sphere and our val split is a permutation of that "
          "same set — held-out views sit INTERSPERSED among training views. We interpolate "
          "exactly as they do. Their SVSS case (single-look, one-sided visibility) is arguably "
          "harder than anything we run.", False)],
        [("::3.  “This literature has no occlusion / absorption term.”  ", True), ("FALSE.", True)],
        [("SH-SAS Eq. 7 is NeRF-style accumulated transmittance, and DART models transmittance "
          "too. Occlusion is a gap in OUR operator, not in theirs.", False)],
        [("::Why put this on a slide", True)],
        [("Every one of these was believed here a week ago. Each was found by reading the "
          "primary source instead of the abstract. Stating them first is cheaper than being "
          "corrected in review — and what remains after they are removed is what we can defend.", False)],
    ], top=1.30, size=12.5)

    # -- what survives -------------------------------------------------------
    s = content_slide(prs, "What survives the retractions — and it is enough")
    bullets(s, [
        [("::Ours, verified against primary sources", True)],
        [("Angular order ", False), ("DERIVED, not searched", True),
         (": L ≤ 2k·a, with the measurement (49→16 coefficients, 5.42M→1.77M params, "
          "val 38.5% → 25.3%) and the corollary that finer grids need LOWER degree.", False)],
        [("Model order ", False), ("countable and bracketable", True),
         (": 4 000 voxels BREAKS the fit, 20 000 works. Structurally unstateable in a hash grid.", False)],
        [("The ", False), ("view-density / sampling law", True),
         (" (n = 100/200/400/800/1800). SH-SAS varies view count at ONE point, qualitatively. "
          "No radar-NVS paper reports a curve.", False)],
        [("Opacity that is ", False), ("regime-neutral", True),
         (": Σ|c_lm|² is rotation-invariant, so it is indifferent to whether the lobe is broad "
          "(diffuse) or narrow (specular). Their |σ_DC| keying assumes the diffuse end.", False)],
        [("::The physical argument behind that last one", True)],
        [("A flat conducting plate is ", False), ("perfectly opaque and has almost no isotropic "
          "return", True), (" — tie opacity to the DC coefficient and occluders go transparent "
          "exactly where occlusion matters. We already learned this: DC-based PRUNING deleted "
          "the specular scatterers a PEC target is made of.", False)],
        [("::And still ours", True)],
        [("No surface prior of any kind — no SDF, no normals, no Lambertian, no TV.", True)],
    ], top=1.25, size=12)

    # -- Reed unlock ---------------------------------------------------------
    s = content_slide(prs, "The unlock: SH-SAS’s code is not out — but their baseline’s is")
    bullets(s, [
        [("::Status of the competitor’s code", True)],
        [("“We will release our implementation as open source after peer review.” "
          "No repository, no project page, no URL anywhere in the paper or supplement. "
          "Now published (IEEE Xplore 11533303) — ", False),
         ("re-check before reimplementing", True), (".", False)],
        [("::Reed et al. 2023 IS public — and it is the baseline in every SH-SAS table", True)],
        [("ACM TOG 42(4) 2023 · github.com/awreed/Neural-Volumetric-Reconstruction-for-Coherent-SAS "
          "· shares two authors with SH-SAS · runs on AirSAS.", False)],
        [("It supplies three things at once: the ", False), ("AirSAS data pipeline", True),
         (", the ", False), ("pulse deconvolution", True),
         (" SH-SAS is layered on top of, and a ", False), ("calibrated baseline", True), (".", False)],
        [("::Chain the comparison", True)],
        [("Their margin over Reed is published. Our margin over Reed on the same data is "
          "therefore directly comparable — ", False),
         ("without needing their code at all", True), (".", False)],
        [("::And we still reimplement them, per Daqian", True)],
        [("Both routes. A reimplemented competitor alone is discounted by reviewers; chained "
          "through a public baseline it is corroborated.", False)],
    ], top=1.28, size=12.5)

    # -- SpINR: the work we had missed ---------------------------------------
    s = content_slide(prs, "The audit’s biggest surprise: SpINR was uncatalogued, and it is the closest work")
    bullets(s, [
        [("SpINR", True), (", “Neural Volumetric Reconstruction for FMCW Radars” "
          "(arXiv 2503.23313) — cited by SH-SAS as [49], never assessed here until today.", False)],
        [("::What it shares with us — almost everything", True)],
        [("Static target on a ", False), ("turntable with a cylindrical inverse synthetic "
          "aperture", True), (", monostatic. That is our experimental geometry.", False)],
        [("Supervises on the ", False), ("COMPLEX beat signal — real and imaginary parts", True),
         (". They even ablate magnitude-vs-complex supervision.", False)],
        [("Forward model  σ(x) / (N·R_T·R_R) · e^{iφ(x)}  — ", False),
         ("this is our operator with range_model = product", True),
         (". Independent arrival at the same kernel: further confirmation the operator is not a "
          "contribution.", False)],
        [("::What it does NOT do — and this is our opening", True)],
        [("Per-point output is an ", False), ("isotropic σ(x) — no angular model at all", True),
         (". No occlusion (they name it as future work, unimplemented).", False)],
        [("NO NOVEL-VIEW SYNTHESIS.", True),
         ("  “All evaluation uses measurements from the cylindrical aperture… No text discusses "
          "synthesizing signals at novel poses.” They had coherent data in hand and did not do it.", False)],
        [("::Why this is the best single comparison available in radar", True)],
        [("Their isotropic σ(x) against our angular SH model on ", False),
         ("identical physics", True), (" — one variable. And we add the NVS they never did. "
          "Simulated from standard meshes on a turntable, so the protocol is reproducible even "
          "though their data and code are not released.", False)],
    ], top=1.25, size=12)

    # -- the baseline suite --------------------------------------------------
    s = content_slide(prs, "The baseline suite: five competitor methods, reimplemented, on our data")
    bullets(s, [
        [("::The rule (Daqian): reimplement and test on our setup — missing code is not a blocker", True)],
        [("Our data becomes the common ground, and that is an ", False), ("advantage", True),
         (": we hold exact ground truth, stronger than the LiDAR / Scaniverse truth most of them "
          "score against.", False)],
        [("::Ordered by value / cost", True)],
        [("1.  SpINR", True), (" — isotropic σ(x) + our exact operator. ", False),
         ("We may already have it", True), (": --scene-repr grid, degree 0, range_model=product.", False)],
        [("2.  SH-SAS", True), (" — --scene-repr hash_sh + --opacity-key dc. The named competitor.", False)],
        [("3.  Radar Fields", True), (" — occupancy × reflectance, FFT-POWER supervision. "
          "The power-only family’s representative, and static. ", False),
         ("This is where “compare scene reconstruction against power-only methods” executes.", True)],
        [("4.  Sugavanam & Ertin", True), (" — scattering centres → SDF. The surface-prior "
          "control, settling the standing no-SDF question as a BASELINE, not as our method.", False)],
        [("5.  SAR-GS", True), (" — 3D Gaussians + SH on backprojected amplitude images. "
          "⚠ conflicts with the standing “no Gaussians” preference — fine as a baseline, "
          "but flagging it.", False)],
        [("::Making a reimplementation credible", True)],
        [("Validate each port against the original’s own published numbers on its own public "
          "benchmark FIRST, then run the validated port on our data. Available for SH-SAS and "
          "Reed (AirSAS) and SAR-GS (MSTAR). ", False),
         ("Not available for Radar Fields — say so in the paper rather than leaving it implicit.", True)],
    ], top=1.25, size=12)

    # -- SpINRv2 full read ---------------------------------------------------
    s = content_slide(prs, "SpINRv2 (Aug 2025) — full read. It may have diagnosed OUR sphere defects.")
    bullets(s, [
        [("::Their forward model is nearly ours — with one piece done better", True)],
        [("Z_k = ∫ σ(x)/(N·R_T·R_R) · e^{i2πf₀τ} · (1−e^{i2πSτN})/(1−e^{i(2πSτ−β_k)}) dx  — the last "
          "factor is a ", False), ("closed-form Dirichlet / DFT-leakage kernel", True),
         (". Our range_operator.py solves the same “scatterers are not at bin centres” problem with "
          "NUFFT gridding + deapodization — ", False), ("theirs is exact, ours is approximate", True),
         (". Worth evaluating a swap.", False)],
        [("::SUB-BIN AMBIGUITY — and we are 19× deeper into it than anything they tested", True)],
        [("Within one bin Δr = c/2B, many sub-bin positions give near-identical phase. The start "
          "frequency adds a GLOBAL phase 2πf₀τ, so higher f₀ ⇒ more configurations with the same "
          "spectrum. Dangerous when λ/4 < Δr. Their symptoms: ", False),
         ("“faint shells around the true geometry”, duplication, blur.", True)],
        [("Their worst case (f₀ = 5 GHz):  λ/4 = 15 mm vs Δr = 42 mm → 1 : 2.8.        ", False),
         ("RIFT (79 GHz, B = 3 GHz):  λ/4 = 0.95 mm vs Δr = 50 mm → 1 : 52.", True)],
        [("::Candidate re-diagnosis of two of our standing defects", True)],
        [("We attributed the ", False), ("+3.5 cm shell bias", True),
         (" to “init-inheritance” and the ", False), ("~2× radial smear", True),
         (" to resolution budget. But a flat, multi-modal sub-bin loss landscape ", False),
         ("IS what init-inheritance looks like from the outside", True),
         (" — if many sub-bin radii are near-degenerate, the optimiser stays where it started. "
          "These may be one phenomenon, not two — and a physics explanation, not an accident.", False)],
        [("Their disambiguators: multi-view consistency (“only the true surface satisfies the "
          "response across ALL transmitter positions”) + smoothness/sparsity priors. We have 750+ "
          "views — ", False), ("checking whether that already suppresses it is cheap and high-value.", True)],
        [("::Also actionable: staged supervision", True)],
        [("Their perturbation study — magnitude loss has strong gradients at ~10 cm scales; real/"
          "imaginary only become discriminative at ~1 mm. So they run magnitude-only first, then "
          "combined. ", False), ("A complex loss from scratch is poorly conditioned at coarse scales.", True)],
    ], top=1.15, size=11)

    s = content_slide(prs, "SpINRv2 — what they report, and what they still do not do")
    table(s, [
        ("Method", "IoU ↑", "Chamfer ↓", "Hausdorff ↓", "PSNR ↑", "SSIM ↑", "LPIPS ↓"),
        ("SpINRv2", "0.0908", "0.0055", "0.0713", "17.06", "0.801", "0.248"),
        ("TF-TS (time-domain fwd + temporal sup.)", "0.0483", "0.0219", "0.1470", "9.27", "0.469", "0.683"),
        ("TF-SS (time-domain fwd + spectral sup.)", "0.0177", "0.0100", "0.0722", "13.51", "0.719", "0.392"),
        ("RQ (range quantization)", "0.0135", "0.0728", "0.1856", "6.27", "0.390", "0.806"),
        ("Coherent backprojection", "0.0598", "0.0099", "0.0461", "11.28", "0.691", "0.426"),
    ], top=1.25, width=12.4, col_widths=(4.0, 1.4, 1.5, 1.6, 1.4, 1.3, 1.2),
       header_size=10.5, body_size=10)
    bullets(s, [
        [("They claim “best across all six metrics” — ", False),
         ("their own table contradicts it: classical backprojection wins Hausdorff", True),
         (" (0.0461 vs 0.0713). And note the absolute scale: IoU ≈ 0.09 for everyone.", False)],
        [("::What they still do NOT do — all of it is our space", True)],
        [("No angular / view-dependent model", True), (" (isotropic σ(x)).  ", False),
         ("No occlusion.", True), ("  Simulated only, no real data.  ", False),
         ("NO NOVEL-VIEW SYNTHESIS", True), (" — never mentioned in the full text.  Not "
          "prior-free either: smoothness + sparsity.  Experiments convert multistatic → MONOSTATIC.", False)],
        [("::And they hand us the contribution, verbatim (their §4.2)", True)],
        [("“This formulation can be extended to incorporate … ", False),
         ("transmission attenuation models", True), (" …, ", False),
         ("scattering probability as a function of incident angle or local geometry", True),
         (" (e.g., Lambertian, specular, or volumetric models), and multipath interference or ", False),
         ("occlusion", True), (".”  Both of our closest competitors name our contribution as "
          "their future work — SH-SAS names radar, SpINRv2 names angular scattering and occlusion.", False)],
        [("::Validation targets for our port (their Table 2, per-object Chamfer)", True)],
        [("bunny 0.0050 · spot 0.0066 · lucy 0.0044 · armadillo 0.0042 · dragon 0.0042 · "
          "woody 0.0061 · teapot 0.0080 — standard meshes, so reproducible without their data.", False)],
    ], top=3.55, size=11)

    # -- completeness, and every baseline is an ablation ---------------------
    s = content_slide(prs, "Not novelty — COMPLETENESS. And every baseline is an ablation of our model.")
    table(s, [
        ("forward-model axis", "who else has it"),
        ("coherent (complex) forward model", "SpINR, SH-SAS, Reed only"),
        ("exact near-field, no far-field approximation", "those three (Sugavanam & Ertin use a non-uniform 3D FFT)"),
        ("true bistatic MIMO, per-element positions", "RIFT and SpINRv2 (v1 converts multistatic → monostatic)"),
        ("view-dependent angular model", "RIFT, SH-SAS (SpINR and Reed are isotropic)"),
        ("angular order DERIVED from physics", "RIFT ALONE"),
        ("no surface / material prior", "RIFT ALONE"),
        ("occlusion / visibility", "SH-SAS, DART, RadarSim, GeRaF — NOT US"),
    ], top=1.20, width=12.4, col_widths=(5.0, 7.4), header_size=11, body_size=10)
    bullets(s, [
        [("Most complete coherent forward model in the set on every axis except OCCLUSION — "
          "which is the single addition that makes it strictly dominant.", True)],
        [("::Every baseline is a LESION of our model — so the baseline suite IS the ablation study", True)],
        [("SpINR", True), (" = ours minus the angular model, minus true bistatic.   ", False),
         ("Reed", True), (" = ours minus the angular model.   ", False),
         ("Radar Fields", True), (" = ours minus coherence.", False)],
        [("SH-SAS", True), (" = ours with diffuse-keyed opacity instead of regime-neutral.   ", False),
         ("Sugavanam & Ertin", True), (" = ours plus a surface prior, minus complex amplitude.", False)],
        [("Every ablation row has a ", False), ("real published paper behind it", True),
         (" — which is exactly why a reviewer cannot call them strawmen. The contribution becomes "
          "“we assemble the most complete coherent forward model and measure what each element "
          "buys”, which survives the concession that no single element is novel.", False)],
        [("::Two caveats to state rather than hide", True)],
        [("Completeness ≠ fidelity: on diffuse sonar targets SH-SAS’s Lambertian + DC-opacity may "
          "be MORE accurate despite being more assumption-laden. And within the class everyone — "
          "us included — omits polarization (EM is vector), multiple scattering and diffraction.", False)],
    ], top=4.30, size=11.5)

    # -- the PEC problem -----------------------------------------------------
    s = content_slide(prs, "The PEC problem — the biggest threat to our own positioning")
    bullets(s, [
        [("::The objection, stated at its strongest", True)],
        [("Every dataset we own is PEC. PEC is an idealization used to make EM simulation "
          "tractable; ", False), ("no real target is one", True),
         (". A method that beats the incumbents only under a condition that is never true in "
          "reality is not a contribution, and a reviewer will say so in one line.", False)],
        [("Worse: “opacity ≠ isotropic amplitude” is ", False), ("most extreme for PEC", True),
         (" — so PEC-only evidence proves it in the one regime engineered to make it true.", False)],
        [("::The reframe that survives", True)],
        [("The axis is not PEC-vs-real. It is ", False),
         ("where the target sits on the specular↔diffuse continuum", True),
         (" — continuous, set by surface roughness relative to λ, varying WITHIN one object "
          "(smooth panels vs. rough tires vs. cavities).", False)],
        [("At 79 GHz (λ ≈ 3.8 mm) vehicle panels, aircraft skin, signage and walls ARE optically "
          "smooth. The specular end is not a fiction — it is just not the whole story.", False)],
        [("::AirSAS discharges half of it for free", True)],
        [("Real measured data on DIFFUSE 3-D printed targets — the opposite end of the "
          "continuum. A win on AirSAS + a win on our PEC radar data ", False),
         ("brackets the continuum from both ends with no new simulation", True), (".", False)],
        [("Still owed: non-PEC radar data, because with max_trans = 0 occlusion is BINARY and "
          "the extinction MAGNITUDE is unidentifiable — a graded transmittance cannot be fitted "
          "on any dataset we currently own. But that is off the critical path.", False)],
    ], top=1.22, size=11.5)

    # -- how we build it -----------------------------------------------------
    s = content_slide(prs, "How we build it: their method as ARMS INSIDE RIFT, not a separate port")
    bullets(s, [
        [("::The design", True)],
        [("train.py already dispatches on --scene-repr {mlp, grid, grid_sh, point_sh} over a "
          "common active_scatterers(dθ, dφ) interface. SH-SAS decomposes into two additions "
          "that slot straight in:", False)],
        [("--scene-repr hash_sh", True),
         (" — multi-resolution hash encoding (16 levels → 4096) + MLP 2×32 → 2(L+1)² channels. "
          "Pure PyTorch; no tinycudann needed at our scale.", False)],
        [("--opacity-key {dc, energy}", True),
         (" — an occlusion module in the operator. ‘dc’ reproduces theirs (ρ̂ = |σ_DC|·ζ, "
          "normals from ∇|σ_DC|, Lambertian lobe); ‘energy’ is ours.", False)],
        [("::Why this beats a standalone reimplementation", True)],
        [("Operator, data, split, optimizer and training loop are held ", False),
         ("identical", True), (", so representation / order-selection / opacity vary ", False),
         ("one at a time", True), (". It answers the reviewer’s question — “is the win the "
          "representation or the order selection?” — ", False),
         ("by construction rather than by argument", True), (". And it is less work.", False)],
        [("::The caveat we must state in the paper", True)],
        [("This is their REPRESENTATION and OPACITY MODEL inside our operator — not their full "
          "pipeline (ellipsoidal sampling, pulse deconvolution, TV priors). For AirSAS we use "
          "their real pipeline pieces via Reed’s public repo. For radar, this port IS the fair "
          "comparison — and is precisely what their own “swap in a modality-specific forward "
          "model” future work describes.", False)],
    ], top=1.25, size=12)

    # -- sequencing ----------------------------------------------------------
    s = content_slide(prs, "Sequencing — four gates")
    bullets(s, [
        [("::Gate S1 — reads and plumbing (analysis agent, days)", True)],
        [("Is suppl. Table 4 on SIMULATED or AirSAS Armadillo?", True),
         (" Their simulator is unreleased too — if simulated, Table 4 is a reference point, not "
          "a target, and the comparison re-anchors on AirSAS via Reed. ", False),
         ("This is a read, and it is blocking.", True)],
        [("Pull AirSAS + Reed’s repo; reproduce Reed’s published result; build the AirSAS → RIFT "
          "loader; ", False), ("recover the amplitude normalization", True),
         ("; pass the per-view coherence gate. Read SpINR in parallel.", False)],
        [("::Gate S2 — the build (analysis agent)", True)],
        [("hash_sh + the opacity module. Re-check for an SH-SAS code release FIRST.", False)],
        [("::Gate S3 — the head-to-head (experiment manager, GPU)", True)],
        [("{hash_sh, grid_sh} × {L fixed 3, L capped} × {opacity dc, energy, none}, on AirSAS "
          "AND our radar data. ", False), ("This is the paper.", True)],
        [("::Gate S4 — beyond submission 1", True)],
        [("Non-PEC data generation so occlusion becomes identifiable; then the specular↔diffuse "
          "ladder.", False)],
        [("::Minimum submittable set", True)],
        [("S1 + S2 + S3 — a head-to-head win on AirSAS and on our radar data. Everything else "
          "is supporting analysis.", True)],
    ], top=1.25, size=12)

    # -- risks ---------------------------------------------------------------
    s = content_slide(prs, "What would kill this, and what we do about it")
    bullets(s, [
        [("We lose on AirSAS.", True),
         (" Our differentiators are specular-motivated; AirSAS targets are DIFFUSE — the regime "
          "where their DC-keyed opacity and Lambertian lobe are CORRECT. ", False),
         ("This is the real risk.", True),
         (" Mitigation: it is also the reason to run it early and cheaply rather than argue "
          "about it. And the L = 3 prediction is informative even if we lose on error.", False)],
        [("Table 4 turns out to be unreproducible.", True),
         (" If it is on their unreleased simulator, we cannot match those numbers at all. "
          "Mitigation: re-anchor everything on Reed, whose code is public. Gate S1 settles it "
          "before any build starts.", False)],
        [("The normalization is unrecoverable.", True),
         (" Their L1/MSE have no stated scale. Mitigation: Reed’s pipeline defines the "
          "preprocessing; if it still cannot be pinned, we publish the normalized metric "
          "ourselves and report both.", False)],
        [("SH-SAS extends to radar before we submit.", True),
         (" Their stated future work. Mitigation: the port is NOT a swap — opacity, normals and "
          "the reflectance lobe are all keyed to diffuse physics and do not survive the "
          "specular regime. That is a technical head start, not a rhetorical one.", False)],
        [("SpINR already did it.", True),
         (" Unassessed FMCW-radar volumetric reconstruction. Mitigation: read it this week, "
          "before the build.", False)],
    ], top=1.35, size=12.5)


# --------------------------------------------------------------------------
# HISTORICAL SNAPSHOT -- Round 6 as of 2026-08-06. Preserved for provenance but
# no longer called by main(); add_part3_round6 below is the audited final section.
#
# Numbers: scripts/scrape_training_logs.py (val/train), the runs' own printed
# zeta, and scripts/eval_b787_energy_on_airframe.py (support test).
# Figures:  plot_val_error_curves.py --set round6, render_recon_grid.py --set round6.
# --------------------------------------------------------------------------

R6_ARMS = [
    ("arm", "what it adds", "decides"),
    ("O1 · ζ frozen 0.1", "--occlusion --occlusion-freeze-scale --occlusion-scale 0.1",
     "does a nearly-transparent visibility term help at all"),
    ("O2 · ζ frozen 1.0", "--occlusion --occlusion-freeze-scale --occlusion-scale 1.0",
     "the optical-depth ladder — how opaque is too opaque"),
    ("O3 · ζ learned", "--occlusion --occlusion-lr 3e-3   (energy-keyed)",
     "let the data choose the optical depth; ζ itself is the measurement"),
    ("O4 · ζ learned, DC-keyed", "--occlusion --opacity-key dc --occlusion-lr 3e-3",
     "SH-SAS's ρ̂ = |σ_DC|·ζ as an ARM — energy- vs DC-keying (S6)"),
    ("S1 · magnitude λ=2", "--mag-weight 2.0",
     "the magnitude term alone"),
    ("S2 · magnitude λ=2 + warm-up", "--mag-weight 2.0 --mag-warmup-epochs 15",
     "the term vs the STAGING — SpINRv2's actual recipe"),
]

R6_RESULTS_20260806_SNAPSHOT = [
    ("arm", "epoch", "train rel-MSE ↓", "best val ↓", "baseline val, same epoch", "ζ"),
    ("baseline · deg 3, no occlusion", "150 / 150", "0.98%", "26.1%", "—", "—"),
    ("O1 · occ, ζ frozen 0.1", "83 / 150", "14.09%", "71.2% @69", "32.1%", "0.1 fixed"),
    ("O2 · occ, ζ frozen 1.0", "71 / 150", "99.97%", "100.0% @70", "31.3%", "1.0 fixed"),
    ("O3 · occ, ζ learned (energy)", "63 / 150", "1.04%", "29.0% @62", "32.2%", "0.155 → 5e−15"),
    ("O4 · occ, ζ learned (DC)", "66 / 150", "1.01%", "28.0% @65", "32.1%", "0.135 → 5e−15"),
    ("S1 · magnitude λ = 2", "94 / 150", "0.50%", "37.1% @93", "27.7%", "—"),
    ("S2 · magnitude λ = 2 + warm-up", "93 / 150", "0.53%", "41.5% @92", "27.8%", "—"),
]

R6_SUPPORT_20260806_SNAPSHOT = [
    ("run (best checkpoint)", "epoch", "energy ≤ 5 mm of STL ↑", "concentration ↑", "≤ 10 mm ↑"),
    ("baseline · deg 6 (val 38.5%)", "149", "17.3%", "87×", "35.0%"),
    ("baseline · deg 3 (val 26.1%)", "149", "34.2%", "172×", "59.0%"),
    ("R5 · deg 3 + prune 20k (val 25.3%)", "149", "45.2%", "227×", "73.7%"),
    ("O1 · occ ζ frozen 0.1", "70", "1.2%", "6×", "5.4%"),
    ("O2 · occ ζ frozen 1.0", "71", "0.0%", "0×", "0.0%"),
    ("O3 · occ ζ learned (energy)", "63", "29.9%", "150×", "53.0%"),
    ("O4 · occ ζ learned (DC)", "66", "30.9%", "156×", "53.8%"),
    ("S1 · magnitude λ = 2", "94", "21.8%", "110×", "38.7%"),
]


def add_part3_round6_20260806_snapshot(prs):
    # -- section title ------------------------------------------------------
    s, sub = add_slide(prs, "Title Slide", "Part 3 — Round 6: Occlusion and SpINRv2 Supervision")
    if sub is not None:
        sub.text_frame.text = ("Daqian Bao — August 6, 2026 · the first experiment set off the "
                               "plan · all six arms in flight")

    # -- what changed in the operator ---------------------------------------
    s = content_slide(prs, "What Round 6 changes — and why these two, out of the whole plan")
    bullets(s, [
        [("::1.  Occlusion — the ONE axis where our operator was behind four published methods", True)],
        [("w_n → w_n · exp(−2τ), with τ a ray-marched optical depth from the array phase centre. "
          "Everything else in the operator is unchanged.", False)],
        [("Computed per (point, ", False), ("VIEW", True),
         ("), not per (point, Tx, Rx): the array is 2.85 cm across at 10 m, so the lateral ray "
          "walk 1 m into the scene is 1/22 of a voxel. 256× cheaper, and ", False),
         ("the range factorization survives", True), (".", False)],
        [("Opacity keyed to rotation-invariant angular ENERGY Σ|c_ℓm|², not to the DC "
          "coefficient — ", False), ("this is the S6 claim, now a measurement", True),
         (": a flat conducting plate is perfectly opaque and has almost no isotropic return, "
          "so DC-keying makes occluders transparent exactly where occlusion matters. "
          "--opacity-key dc runs SH-SAS's version as arm O4.", False)],
        [("ζ → 0 recovers the no-occlusion operator EXACTLY, so an occlusion arm is a strict "
          "superset of its baseline — and ζ is gauge-invariant, reading as the one-way optical "
          "depth of one average-energy voxel.", False)],
        [("::2.  SpINRv2's staged magnitude → complex supervision", True)],
        [("Their Fig. 14: the magnitude loss has gradients at ~10 cm scales while Re/Im only "
          "bite at ~1 mm, so a complex loss from scratch is poorly conditioned at coarse "
          "scales. Relevant because our sub-bin ambiguity ratio (λ/4 : Δr = 1:52) is ~19× "
          "deeper than their worst tested case.", False)],
        [("::Two SpINRv2 candidates were settled WITHOUT GPU time", True)],
        [("Their Dirichlet/DFT-leakage kernel does not apply to our data at all (we never "
          "deramp — the simulator emits the frequency response directly), and ", False),
         ("--range-model product is a measured null", True),
         (" (0.000% shape difference at 10 m standoff).", False)],
    ], top=1.20, size=12)

    # -- the arms -----------------------------------------------------------
    s = content_slide(prs, "Six arms, one variable each")
    table(s, R6_ARMS, top=1.45, width=12.4,
          col_widths=(2.9, 5.0, 4.5), header_size=12, body_size=11)
    bullets(s, [
        [("All six sit on R4 arm A's configuration — degree 3, ", False),
         ("no prune, no grow", True),
         (" — so the operator is the single variable. Pruning changes the active count, which "
          "changes the extinction field, so R5's winning prune schedule is deliberately held "
          "back until Arm O reads.", False)],
        [("The control is ", False), ("b787_r4_capdeg3", True),
         (", byte-identical apart from the new flags (same seed, same split, same lr). "
          "None of the six has reached 150 epochs, so every number below is compared "
          "EPOCH-MATCHED against that run.", True)],
    ], top=4.30, size=12.5)

    # -- train + val curves --------------------------------------------------
    s = content_slide(prs, "Round 6: training error and held-out validation error")
    picture(s, os.path.join(BR, "r6_error_curves.png"), top=1.35, width=11.2)
    caption(s, "Left: training rel-MSE (log). Right: validation rel-MSE on the 200 held-out "
               "viewpoints. Dashed grey = the no-occlusion baseline, the epoch-matched control. "
               "Markers are each arm's best epoch so far.",
            top=6.60, size=12, align_center=True)

    # -- the numbers ---------------------------------------------------------
    s = content_slide(prs, "Round 6 numbers — every arm still running")
    table(s, R6_RESULTS_20260806_SNAPSHOT, top=1.32, width=12.2,
          col_widths=(3.4, 1.5, 1.8, 1.6, 2.2, 1.7), body_size=12)
    bullets(s, [
        [("::Occlusion", True)],
        [("A frozen optical depth is strictly harmful, and monotonically so: ζ = 0.1 reads "
          "71.2% against a 32.1% control, ζ = 1.0 walks to the ", False),
         ("predict-zero floor", True),
         (" — exp(−2τ) saturates, and a saturated scene has no gradient left to walk back.", False)],
        [("Both LEARNED arms drive ζ from ~0.15 to ", False), ("5e−15 within ~10 epochs", True),
         (" — the data switches the visibility term OFF. Since ζ → 0 recovers the baseline "
          "operator exactly, their 3–4 pt lead at matched epochs is an artifact of the "
          "transient early opacity, ", False), ("not evidence that occlusion helps", True), (".", False)],
        [("O4 (DC-keyed, SH-SAS's rule) is not distinguishable from O3 (energy-keyed) here — "
          "on PEC the extinction COEFFICIENT is unidentifiable (max_trans = 0), so ", False),
         ("S6 cannot be settled on this data", True), (". It needs non-PEC targets.", False)],
        [("::SpINRv2 supervision", True)],
        [("The magnitude term fits train ", False), ("harder", True),
         (" than the baseline (0.50% vs 1.09%) and generalizes ", False), ("worse", True),
         (" (37.1% vs 27.7%). Staging makes it worse still (41.5%). On our data the complex "
          "loss is not the ill-conditioned one — this looks like the same over-fitting axis "
          "the degree ladder measures.", False)],
    ], top=4.05, size=12)

    # -- rendered ------------------------------------------------------------
    s = content_slide(prs, "Round 6, rendered — dense trilinear interpolation vs the STL")
    picture(s, os.path.join(BR, "r6_grid.png"), top=1.45, width=12.6)
    caption(s, "48³ SH grid trilinearly upsampled 4× to 192³, max-intensity projection of "
               "Σ|c_ℓm|², cyan = ground-truth STL. Visualization only — never in the training "
               "path. The two learned-ζ arms are visually the baseline, which is what ζ → 0 "
               "predicts; ζ = 0.1 fragments the support and ζ = 1.0 throws it off the airframe "
               "entirely.",
            top=6.10, size=12, align_center=True)

    # -- the support test ----------------------------------------------------
    s = content_slide(prs, "The falsifiable prediction this round tests — and it FAILS")
    caption(s, "T2 predicted that a scatterer field with no visibility term buys occlusion out "
               "of interference, placing canceling contributions off-target — so adding the term "
               "should move energy back ONTO the airframe.", 1.14)
    table(s, R6_SUPPORT_20260806_SNAPSHOT, top=1.50, width=11.8,
          col_widths=(4.2, 1.4, 2.6, 2.0, 1.6), body_size=12)
    bullets(s, [
        [("Energy = Σ|c_ℓm|² per voxel; concentration = energy fraction ÷ volume fraction "
          "within the tolerance (5 mm is under one 6.25 mm voxel pitch). Ranking is stable "
          "at both tolerances.", False)],
        [("::The read", True)],
        [("No occlusion arm improves on-airframe support. The frozen arms move energy "
          "decisively OFF it (1.2% and 0.0%), and the learned arms sit at 30% against the "
          "baseline's 34% — ", False), ("the T2 mechanism is not confirmed", True), (".", False)],
        [("What DOES track on-airframe energy is capacity and support control: "
          "17.3% (deg 6) → 34.2% (deg 3) → ", False), ("45.2% (deg 3 + prune to 20k)", True),
         (", which is also the val ranking. The same lever moves both.", False)],
        [("Caveat stated rather than hidden: the baseline rows are at epoch 149 and the R6 "
          "rows at 63–91. The direction is large enough to survive that, but the round has to "
          "finish before this is a result.", False)],
    ], top=4.55, size=12)

    # -- R6 under the geometry metrics ---------------------------------------
    s = content_slide(prs, "Round 6 on the competitors' geometry metrics")
    table(s, GEOM_R6_20260806_SNAPSHOT, top=1.35, width=12.2,
          col_widths=(3.6, 1.6, 1.6, 1.6, 1.5, 1.1, 1.2),
          header_size=11, body_size=11)
    bullets(s, [
        [("The two collapsed arms are unambiguous here: ζ = 1.0 reads ", False),
         ("F1 = 0.000 and a Chamfer 700× the baseline's", True),
         (" — the geometry metric and the signal metric agree completely when a run "
          "actually fails.", False)],
        [("The four healthy arms are ", False),
         ("geometrically indistinguishable from the baseline", True),
         (" (CD 4.99–5.21e−05, F1 0.843–0.859) while their val rel-MSE spans 26–42%. "
          "Same point as the previous section, now within one round: at each arm's own "
          "best threshold, geometry cannot see the difference that novel-view synthesis "
          "measures.", False)],
        [("The fixed-threshold column does see it — 0.423 (baseline) vs 0.311 / 0.276 "
          "for the two magnitude-supervision arms, in the same order as their val error.", False)],
    ], top=4.15, size=12)

    # -- verdict -------------------------------------------------------------
    s = content_slide(prs, "Round 6: reading so far, and what it changes")
    bullets(s, [
        [("::What is already decided", True)],
        [("Occlusion, as an occlusion, is a ", False), ("clean negative on this data", True),
         (" — not a broken build. ζ is learnable, gauge-invariant, and the optimizer sends it "
          "to zero. The term is now in the operator and costs nothing when it is not wanted.", False)],
        [("That is the expected outcome for PEC: max_trans = 0 means occlusion is BINARY, so "
          "the extinction magnitude is unidentifiable. ", False),
         ("The build was still worth it", True),
         (" — it closes the one capability gap against SH-SAS/DART/RadarSim/GeRaF, and it is a "
          "prerequisite for AirSAS and for any non-PEC data.", False)],
        [("SpINRv2's magnitude supervision does not transfer. Reported rel-MSE stays data-only "
          "complex, so the arms remain comparable — and they lose.", False)],
        [("::What it does NOT decide", True)],
        [("Energy- vs DC-keyed opacity (S6). Both arms drove ζ to zero, so the keying was "
          "never exercised. ", False), ("This needs non-PEC data — it is now the concrete "
          "argument for generating it", True), (".", False)],
        [("::Next", True)],
        [("Let the six arms finish (they are cheap now), then: Gate S1 of the plan — pull "
          "AirSAS + Reed's public repo, recover the amplitude normalization, and get a number "
          "on somebody else's benchmark. That is what Part 2 says the paper is.", True)],
    ], top=1.45, size=13.5)


R6_SIGNAL_FINAL = [
    ("arm", "train rel-MSE ↓", "best val rel-MSE ↓", "final ζ", "result"),
    ("control · no occlusion", "0.9767%", "26.0992%", "—", "clean control"),
    ("O1 · frozen ζ=0.1", "9.2118%", "66.6597%", "0.1", "harmful"),
    ("O2 · frozen ζ=1.0", "99.9817%", "100.0005%", "1.0", "collapsed"),
    ("O3 · learned, energy", "0.9512%", "25.2198%", "5.36e−16", "opacity off"),
    ("O4 · learned, DC", "0.9494%", "25.0540%", "5.39e−16", "BEST NVS"),
    ("S1 · magnitude λ=2", "0.4597%", "32.4232%", "—", "overfits"),
    ("S2 · magnitude + warm-up", "0.4692%", "34.2174%", "—", "overfits more"),
]

R6_GEOM_FIXED_FINAL = [
    ("arm · fixed t=0.20", "Chamfer m² ↓", "HD95 mm ↓", "IoU ↑", "F1 ↑"),
    ("control · no occlusion", "1.7774e−4", "22.084", "0.0782", "0.4225"),
    ("O1 · frozen ζ=0.1", "1.4010e−2", "172.722", "0.0040", "0.0473"),
    ("O2 · frozen ζ=1.0", "3.9451e−2", "202.254", "0.0000", "0.0000"),
    ("O3 · learned, energy", "1.6065e−4", "21.647", "0.0873", "0.4633"),
    ("O4 · learned, DC", "1.5970e−4", "21.641", "0.0878", "0.4688"),
    ("S1 · magnitude λ=2", "7.6070e−4", "26.669", "0.0617", "0.3482"),
    ("S2 · magnitude + warm-up", "1.1132e−3", "106.199", "0.0589", "0.3355"),
]

R6_GEOM_ORACLE_FINAL = [
    ("arm · metric-specific threshold", "best Chamfer ↓", "best HD95 ↓", "best IoU ↑", "best F1 ↑"),
    ("control · no occlusion", "5.136e−5", "10.179", "0.3005", "0.8435"),
    ("O1 · frozen ζ=0.1", "6.106e−3", "158.575", "0.0078", "0.1004"),
    ("O2 · frozen ζ=1.0", "3.6315e−2", "173.781", "0.0000", "0.0000"),
    ("O3 · learned, energy", "5.1687e−5", "11.093", "0.2899", "0.8452"),
    ("O4 · learned, DC", "4.8316e−5", "9.924", "0.2949", "0.8560"),
    ("S1 · magnitude λ=2", "4.9896e−5", "10.390", "0.3116", "0.8621"),
    ("S2 · magnitude + warm-up", "5.0250e−5", "10.390", "0.3102", "0.8601"),
]

AUDIT_STATUS = [
    ("experiment", "artifacts", "scientific status"),
    ("Round 6 RIFT", "7 / 7 final", "REPORTABLE"),
    ("SpINR-style degree 0", "final", "REPORTABLE · adapted ablation"),
    ("Radar Fields", "final + provenance pass", "geometry + matched power signal reportable"),
    ("RadarSplat S0", "3k-step final · E0 pending", "PROVISIONAL NEGATIVE · planar field"),
    ("RIFT+Sonar", "4 / 4 final", "REPORTABLE NEGATIVE"),
    ("independent SH-SAS", "0 / 4 final", "NOT REPORTABLE YET"),
    ("Sugavanam–Ertin", "Stage 1 incomplete", "NOT REPORTABLE YET"),
]

RADAR_SIGNAL_AUDIT = [
    ("method", "coherent complex rel-MSE ↓", "range-power rel-MSE ↓", "comparison status"),
    ("RIFT · learned-DC R6", "25.0540%", range_power_pct("r6"), "best coherent + power NVS"),
    ("RIFT · target-20k R5", "25.3216%", range_power_pct("r5"), "selected sparse RIFT"),
    ("SpINR-style degree 0", "28.5352%", range_power_pct("spinr"), "adapted isotropic ablation"),
    ("RadarSplat S0", "N/A", "92.7748%*", "terminal val · formal E0 pending"),
    ("Radar Fields", "N/A", "304.8836%", "power-only baseline"),
]

RADAR_GEOM_FIXED_AUDIT = [
    ("method · fixed t=0.20", "Chamfer m² ↓", "Hausdorff mm ↓", "HD95 mm ↓", "IoU ↑", "F1 ↑"),
    ("RIFT · learned-DC R6", "1.5970e−4", "51.997", "21.641", "0.0878", "0.4688"),
    ("SpINR-style degree 0", "1.2767e−4", "27.436", "20.956", "0.1058", "0.6052"),
    ("Radar Fields", "1.3873e−2", "234.401", "174.765", "0.0009", "0.0062"),
    ("RadarSplat S0", "N/A", "N/A", "N/A", "N/A", "N/A · planar field"),
]

RADAR_GEOM_ORACLE_AUDIT = [
    ("method · per-metric oracle", "Chamfer ↓", "Hausdorff ↓", "HD95 ↓", "IoU ↑", "F1 ↑"),
    ("RIFT · learned-DC R6", "4.8316e−5 @.60", "20.489 @.40", "9.924 @.60", "0.2949 @.70", "0.8560 @.60"),
    ("SpINR-style degree 0", "9.6892e−5 @.40", "23.676 @.25", "15.725 @.40", "0.1111 @.30", "0.6876 @.30"),
    ("Radar Fields", "4.3883e−3 @.95", "161.083 @.95", "120.614 @.95", "0.0085 @.95", "0.0546 @.95"),
    ("RadarSplat S0", "N/A", "N/A", "N/A", "N/A", "N/A · no threshold contract"),
]

SONAR_RIFT_AUDIT = [
    ("RIFT+Sonar · fixed t=.20", "best val ↓", "final val ↓", "Chamfer m² ↓", "Hausdorff mm ↓", "HD95 mm ↓", "IoU ↑", "F1 ↑"),
    ("Armadillo", "0.9930", "1.0395", "0.01832", "258.36", "197.18", "0.00079", "0.00159"),
    ("Buddha", "0.9703", "1.0349", "0.02887", "281.75", "226.91", "0.00062", "0.00124"),
    ("Bunny", "0.9847", "1.0433", "0.02240", "300.88", "220.85", "0.00141", "0.00282"),
    ("XYZ Dragon", "0.9809", "1.6680", "0.02332", "285.49", "225.67", "0.00121", "0.00241"),
    ("MEAN geometry", "—", "—", "0.02323", "281.62", "217.66", "0.00101", "0.00201"),
]

SONAR_SHSAS_CONTEXT = [
    ("published SH-SAS · context only", "Chamfer ↓", "IoU ↑", "F1 ↑"),
    ("Armadillo", "7.009e−5", "0.444", "0.614"),
    ("Buddha", "9.238e−5", "0.611", "0.756"),
    ("Bunny", "1.255e−4", "0.398", "0.569"),
    ("XYZ Dragon", "6.752e−5", "0.357", "0.526"),
    ("MEAN", "8.887e−5", "0.4525", "0.6163"),
]


def add_radarsplat_s0_slide(prs):
    """Terminal RadarSplat S0 result and direct field diagnostic."""
    s = content_slide(prs, "RadarSplat S0 — complete, but near the predict-zero floor")
    picture(
        s,
        os.path.join(BR, "radarsplat_s0_checkpoint_vs_stl.png"),
        top=1.03,
        width=11.7,
    )
    bullets(s, [
        [("Terminal training-side validation on the frozen 200-view tail: ", False),
         ("92.7748% range-power relative MSE", True),
         (" — only 7.23 points below the 100% zero-predictor floor. Formal E0 is pending.", False)],
        [("The exported field is a broad planar sheet: only ", False),
         ("1.80% of native scene weight lies within 1 cm of the B787 surface", True),
         ("; median Gaussian-to-surface distance is 22.1 cm.", False)],
        [("Adaptation caveat: prune opacity 0.0005 vs paper-profile 0.005; requested SH level 10 "
          "executes at the pinned backend's effective degree 4.", False)],
    ], top=5.03, height=2.15, size=11.25)
    return s


def add_part3_round6(prs):
    """Final Round-6 section; supersedes the August-6 in-flight snapshot above."""
    s, sub = add_slide(prs, "Title Slide", "Part 3 — Round 6: Final Results")
    if sub is not None:
        sub.text_frame.text = ("Daqian Bao — audited August 9, 2026 · all seven B787 runs "
                               "complete at 150 epochs")

    s = content_slide(prs, "Round 6 signal results — all arms complete")
    table(s, R6_SIGNAL_FINAL, top=1.28, width=12.2,
          col_widths=(3.4, 1.8, 2.0, 1.5, 3.5), header_size=11.5, body_size=11.5)
    bullets(s, [
        [("O4 is the best held-out result at ", False), ("25.0540%", True),
         (", narrowly ahead of O3 at 25.2198% and the no-occlusion control at 26.0992%.", False)],
        [("Both learned arms drive ζ to ~5.4e−16: ", False),
         ("the visibility term switches itself off", True),
         (". The small final gain is an optimizer-path effect, not evidence that occlusion helps.", False)],
        [("Frozen opacity is harmful; ζ=1 collapses. Magnitude supervision fits training "
          "harder while worsening validation — a clean over-fitting result.", False)],
    ], top=4.18, size=12.5)

    s = content_slide(prs, "Round 6: complete training and held-out curves")
    picture(s, os.path.join(BR, "r6_error_curves.png"), top=1.32, width=11.2)
    caption(s, "Global complex relative MSE. Markers show the best held-out epoch; all seven "
               "curves now run through epoch 150.", top=6.62, size=12, align_center=True)

    s = content_slide(prs, "Round 6 geometry — fixed normalized threshold t=0.20")
    table(s, R6_GEOM_FIXED_FINAL, top=1.28, width=11.9,
          col_widths=(4.1, 2.1, 1.9, 1.8, 2.0), header_size=11.5, body_size=11.5)
    bullets(s, [
        [("The fixed threshold preserves the signal ranking among the healthy arms: O4/O3 lead, "
          "and both magnitude-loss arms lose. The two frozen-opacity failures are unmistakable.", False)],
        [("This is the deployment-style view: one threshold for every method, chosen without "
          "ground-truth tuning per cell.", True)],
    ], top=4.25, size=12.5)

    s = content_slide(prs, "Round 6 geometry — each metric gets its own oracle threshold")
    table(s, R6_GEOM_ORACLE_FINAL, top=1.28, width=12.0,
          col_widths=(4.2, 2.0, 1.9, 1.9, 2.0), header_size=11.5, body_size=11.5)
    bullets(s, [
        [("Oracle tuning makes the healthy fields look nearly interchangeable and even gives "
          "the magnitude-loss arms the highest F1. That contradicts their much worse NVS error.", False)],
        [("Report this competitor convention only beside the fixed-threshold table: each cell "
          "uses ground truth to select a different level set.", True)],
    ], top=4.25, size=12.5)

    s = content_slide(prs, "Round 6, rendered after completion")
    picture(s, os.path.join(BR, "r6_grid.png"), top=1.45, width=12.6)
    caption(s, "Final/best checkpoints on the native 48³ SH grid, trilinearly upsampled only for "
               "visualization. Cyan = B787 STL truth.", top=6.12, size=12, align_center=True)

    s = content_slide(prs, "Round 6 final read")
    bullets(s, [
        [("::Signal outcome", True)],
        [("Best coherent NVS = ", False), ("25.0540%", True),
         (". Learned visibility converges back to no visibility; fixed visibility fails.", False)],
        [("SpINRv2-style magnitude supervision does not transfer: 0.46% train / 32.42% val "
          "without warm-up and 0.47% / 34.22% with warm-up.", False)],
        [("::Geometry outcome", True)],
        [("Fixed-threshold geometry agrees with signal quality. Per-metric oracle thresholds "
          "erase most healthy-arm differences and must be labelled ground-truth tuned.", False)],
        [("::Scientific conclusion", True)],
        [("Occlusion is a clean negative on this PEC data. Energy-vs-DC opacity remains "
          "unidentified because both learned arms set ζ to zero; test it on non-PEC data.", True)],
    ], top=1.45, size=14)


def add_part4_audited_comparisons(prs):
    r6_vs_spinr = range_power_reduction("r6", "spinr")
    if RANGE_POWER_RESULTS["r6"] is not None:
        r6_vs_radar_fields = 100.0 * (
            1.0 - RANGE_POWER_RESULTS["r6"]["normalized_range_power_rel_mse"] / 3.0488364040713822
        )
    else:
        r6_vs_radar_fields = None

    s, sub = add_slide(prs, "Title Slide", "Part 4 — Audited Radar and Sonar Comparisons")
    if sub is not None:
        sub.text_frame.text = ("RIFT vs SpINR-style, Radar Fields and SH-SAS · artifact audit "
                               "updated August 20, 2026 with RadarSplat S0")

    s = content_slide(prs, "Audit status — complete is not the same as comparable")
    table(s, AUDIT_STATUS, top=1.35, width=12.1,
          col_widths=(3.2, 3.2, 5.7), header_size=11.5, body_size=10.75)
    bullets(s, [
        [("Radar Fields completed, and the selected RIFT/SpINR checkpoints have now been "
          "rerendered into its exact 60 dB-normalized FFT range-power domain.", True)],
        [("The four RIFT+Sonar cells completed and failed scientifically. The controlled "
          "independent SH-SAS cells are unfinished, so no apples-to-apples sonar claim exists yet.", False)],
        [("RadarSplat S0 is operationally complete, but its formal E0 completion artifact is "
          "still absent; keep its 92.7748% terminal validation explicitly provisional.", True)],
    ], top=4.08, size=11.75)

    s = content_slide(prs, "Radar novel-view synthesis — RIFT wins both signal domains")
    table(s, RADAR_SIGNAL_AUDIT, top=1.45, width=12.2,
          col_widths=(3.3, 2.8, 2.6, 3.5), header_size=11.5, body_size=12)
    bullets(s, [
        [("RIFT improves over the SpINR-style degree-0 ablation by ", False),
         ("3.4812 percentage points", True), (", a ", False),
         ("12.20% relative error reduction", True), (".", False)],
        [("Call it a ", False), ("SpINR-style isotropic ablation", True),
         (", not an official reproduction: it shares our exact data, split and forward operator.", False)],
        [(f"Matched range power: RIFT R6 {range_power_pct('r6')} vs SpINR-style "
          f"{range_power_pct('spinr')} — {reduction_text(r6_vs_spinr)} relative reduction. "
          f"Round 5 reverses behind SpINR at {range_power_pct('r5')} despite its better coherent score.", True)],
        [(f"Against the Radar Fields power-only result (304.8836%), RIFT R6 reduces the same "
          f"normalized range-power error by {reduction_text(r6_vs_radar_fields)}.", False)],
        [("RadarSplat S0 reaches 92.7748% on the same frozen validation tail: RIFT R6 is "
          "8.01x lower-error. The asterisk remains until formal E0 rerenders the final checkpoint.", True)],
    ], top=3.55, size=11.5)

    add_radarsplat_s0_slide(prs)

    s = content_slide(prs, "B787 NVS lives on a Fibonacci viewing sphere")
    picture(s, os.path.join("figures", "b787_range_power", "fibonacci_sampling.png"),
            top=1.35, width=12.1)
    bullets(s, [
        [("Blue overlays the 1,800 measured training directions; orange marks the 200 held-out "
          "directions interleaved across the same Fibonacci sphere.", False)],
        [("The dense reference uses measured signal at all 2,000 views. On the next slide, held-out "
          "measurements are replaced by model predictions; metrics remain validation-only.", True)],
    ], top=5.55, height=1.35, size=11.5)

    s = content_slide(prs, "B787 NVS — train/held-out composite and held-out error")
    picture(s, os.path.join("figures", "b787_range_power", "nvs_sphere_comparison.png"),
            top=1.08, width=11.75)

    s = content_slide(prs, "B787 learned 3D fields — direct visual baseline comparison")
    picture(s, os.path.join(BR, "baseline_fields_all.png"), top=1.18, width=12.65)
    caption(s, "Visualization only: RadarSplat is binned from its explicit Gaussian centres; "
               "all display contrasts are method-normalized.", top=5.92, size=11,
            align_center=True)

    s = content_slide(prs, "Radar reconstruction at one fixed threshold — SpINR leads")
    table(s, RADAR_GEOM_FIXED_AUDIT, top=1.45, width=12.2,
          col_widths=(3.3, 1.9, 1.9, 1.7, 1.7, 1.7), header_size=11, body_size=11.5)
    bullets(s, [
        [("At t=0.20, the SpINR-style field is better than RIFT on all five metrics. "
          "Both beat Radar Fields decisively.", True)],
        [("RadarSplat exports a field, but it is planar and has no registered common-threshold "
          "geometry evaluation; N/A is a missing metric, not a missing checkpoint.", False)],
        [("Protocol: native g48 fields, seed 0, 20k truth-surface + 50k truth-volume points; "
          "radar F1 uses a 6.25-mm surface-distance tolerance.", False)],
    ], top=3.68, size=11.75)

    s = content_slide(prs, "Radar reconstruction after oracle tuning — RIFT leads")
    table(s, RADAR_GEOM_ORACLE_AUDIT, top=1.45, width=12.25,
          col_widths=(3.2, 2.1, 1.8, 1.7, 1.7, 1.75), header_size=10.5, body_size=10.5)
    bullets(s, [
        [("The ranking reverses when each metric chooses its own ground-truth-optimal threshold: "
          "RIFT now wins every column.", True)],
        [("Defensible claim: RIFT has better achievable geometry after calibration, but no "
          "threshold-independent geometry advantage over SpINR. Keep both tables together.", False)],
    ], top=3.35, size=13)

    s = content_slide(prs, "RIFT+Sonar — four complete runs, four failed reconstructions")
    table(s, SONAR_RIFT_AUDIT, top=1.28, width=12.35,
          col_widths=(2.15, 1.25, 1.25, 1.65, 1.7, 1.55, 1.35, 1.45),
          header_size=9.5, body_size=10)
    bullets(s, [
        [("All signal errors remain near the predict-zero floor; mean F1 is 0.0020 and mean "
          "HD95 is 217.66 mm. These are operationally complete negative results.", True)],
        [("Threshold sweeps and best-validation checkpoints do not rescue the result. "
          "Sonar F1 is voxel-overlap F1, not the radar surface-distance F1.", False)],
    ], top=3.55, size=13)

    s = content_slide(prs, "SH-SAS — published context exists; the controlled baseline does not yet")
    table(s, SONAR_SHSAS_CONTEXT, top=1.38, width=8.8, left=2.25,
          col_widths=(3.8, 1.8, 1.6, 1.6), header_size=11.5, body_size=11.5)
    bullets(s, [
        [("Published means: Chamfer 8.887e−5, IoU 0.4525, F1 0.6163. HD95 is not reported.", True)],
        [("Context only: SH-SAS uses a different learned frontend and an incompletely documented "
          "threshold contract. It is not an apples-to-apples row against RIFT+Sonar.", False)],
        [("Controlled independent SH-SAS status: ", False), ("0/4 final", True),
         (". Armadillo stopped near 34.7k/50k steps; the other three are near 25k. "
          "Do not promote partial metrics.", False)],
    ], top=3.85, size=12.5)

    s = content_slide(prs, "Final audited read — what can go into the paper today")
    bullets(s, [
        [("::Report now", True)],
        [("Radar NVS: RIFT 25.0540% vs SpINR-style 28.5352% — a 12.20% relative reduction.", False)],
        [("Radar geometry: fixed threshold favors SpINR; per-metric oracle thresholds favor RIFT. "
          "The threshold reversal is itself a result and must remain visible.", False)],
        [("Radar Fields is poor both geometrically and in the matched normalized range-power "
          "domain; it remains N/A only in the coherent-complex column by construction.", False)],
        [("RadarSplat S0 is a provisional negative: 92.7748% terminal power error and a broad "
          "planar field; formal E0 remains pending.", False)],
        [("RIFT+Sonar is a completed negative result; published SH-SAS is context only.", False)],
        [("::Still required", True)],
        [("Finish formal RadarSplat E0, four controlled SH-SAS cells, and Sugavanam–Ertin Stage 2. "
          "Do not promote the S0 training-side number to an atomic E0 result yet.", True)],
        [("Audit fixes: validation epochs are no longer off by one; the retained Radar Fields "
          "artifact is stable across the accidental duplicate launches; sonar HD95 was recomputed.", False)],
    ], top=1.30, size=12.5)


B787_BEST_AT_A_GLANCE = [
    ("axis", "RIFT current best", "B787 reference", "audited read"),
    ("coherent held-out NVS", "25.0540%", "SpINR-style 28.5352%", "RIFT: 12.20% relative reduction"),
    ("fixed t=.20 HD95", "21.641 mm", "SpINR-style 20.956 mm", "SpINR slightly lower"),
    ("fixed t=.20 F1", "0.4688", "SpINR-style 0.6052", "SpINR leads"),
    ("oracle HD95", "9.924 mm @ .60", "SpINR-style 15.725 mm @ .40", "RIFT leads after GT tuning"),
    ("oracle F1", "0.8560 @ .60", "SpINR-style 0.6876 @ .30", "RIFT leads after GT tuning"),
]

B787_SETUP = [
    ("item", "B787 experiment"),
    ("target", "B787 airframe STL · largest dimension 0.10 m"),
    ("views", "2,000 Fibonacci-sphere views · 1,800 train / 200 held out"),
    ("array", "16 Tx × 16 Rx MIMO · exact per-view element positions"),
    ("waveform", "10 GHz centre · 3 GHz bandwidth · 600 frequencies"),
    ("scene", "±0.15 m box · native 48³ grid · 6.25-mm pitch"),
    ("signal metrics", "global coherent-complex rel-MSE + matched normalized range-power rel-MSE"),
]


def add_current_best_b787_all_models_snapshot(prs):
    """Superseded B787-only draft retained for provenance; not called by main()."""
    s, sub = add_slide(prs, "Title Slide", "RIFT: Current Best B787 Results")
    if sub is not None:
        sub.text_frame.text = ("Daqian Bao — August 9, 2026 · coherent radar novel-view "
                               "synthesis and 3D reconstruction")

    s = content_slide(prs, "Current best B787 result — at a glance")
    table(s, B787_BEST_AT_A_GLANCE, top=1.32, width=12.15,
          col_widths=(2.5, 2.4, 3.4, 3.85), header_size=11.5, body_size=11)
    bullets(s, [
        [("The selected model is Round-6 O4, the learned-DC arm: ", False),
         ("25.0540% coherent held-out error", True), (".", False)],
        [("Its learned opacity converges to ζ=5.39e−16, so the endpoint is effectively the "
          "no-occlusion operator; the gain is an optimization-path effect.", False)],
        [("Threshold choice reverses the geometry conclusion. Fixed t=.20 favors SpINR; "
          "per-metric ground-truth tuning favors RIFT. Both views remain in this deck.", True)],
    ], top=3.72, size=12.5)

    s = content_slide(prs, "B787 experiment and evaluation contract")
    table(s, B787_SETUP, top=1.30, width=11.3, left=1.0,
          col_widths=(2.4, 8.9), header_size=12, body_size=11.5)
    bullets(s, [
        [("Signal score = Σ|prediction−target|² / Σ|target|²; lower is better. "
          "A zero predictor scores 100%.", True)],
        [("Geometry uses the native g48 field, B787 STL truth, 20k surface + 50k volume "
          "samples, seed 0, and a 6.25-mm radar F1 tolerance.", False)],
    ], top=4.25, size=12.5)

    s = content_slide(prs, "What we optimize (1/3): the B787 scene model")
    picture(s, os.path.join(BR, "opt_model_scene.png"), top=1.45, width=11.6)

    s = content_slide(prs, "What we optimize (2/3): the coherent signal model")
    picture(s, os.path.join(BR, "opt_model_signal.png"), top=1.45, width=11.6)

    s = content_slide(prs, "What we optimize (3/3): global gain and objective")
    picture(s, os.path.join(BR, "opt_model_objective.png"), top=1.35, width=11.6)

    s = content_slide(prs, "Round 5: capping angular capacity bought generalization")
    picture(s, os.path.join(BR, "val_error_curves.png"), top=1.30, width=9.0)
    caption(s, "All five B787 runs complete at 150 epochs. The degree-3 + target-20k arm "
               "reached 25.3216% held-out error.", top=6.52, size=12, align_center=True)

    s = content_slide(prs, "Round 5 degree ladder — isotropic is the SpINR-style baseline")
    table(s, R5_LADDER, top=1.35, width=11.8,
          col_widths=(3.2, 1.3, 1.6, 2.4, 1.7, 1.6), body_size=12)
    bullets(s, [
        [("Degree 2 is the best unpruned row at 25.5%; degree 3 gives the physics-capped "
          "control at 26.1%; degree 0 gives the isotropic baseline at 28.5%.", False)],
        [("More coefficients improve training fit while worsening held-out error: the failure "
          "mode is angular over-fitting, not missing spatial capacity.", True)],
    ], top=3.62, size=12.5)

    s = content_slide(prs, "Round 5 pruning — 20k voxels helps; 4k destroys the fit")
    table(s, R5_PRUNE, top=1.28, width=11.8,
          col_widths=(2.4, 1.5, 2.2, 2.2, 1.8, 1.7), body_size=11.5)
    bullets(s, [
        [("Target pruning to 20,000 voxels holds at 25.3216%. The 4,000-voxel and mass arms "
          "reach useful intermediate checkpoints and then degrade to 49.7%.", False)],
        [("The coherent support budget is therefore bracketed between 4k and 20k active voxels.", True)],
    ], top=4.34, size=12.5)

    s = content_slide(prs, "Round 5 best reconstruction — baseline blob to resolved airframe")
    caption(s, "Top: uncapped degree 6, val 38.5%. Bottom: degree 3 + prune to 20k, val 25.3216%.",
            top=1.12, size=12, align_center=True)
    picture(s, os.path.join(BR, "baseline_deg6_dense_overlay.png"), top=1.50, width=8.8)
    picture(s, os.path.join(BR, "r5_target20k_dense_overlay.png"), top=4.45, width=8.8)

    s = content_slide(prs, "Round 6 signal results — current best reaches 25.0540%")
    table(s, R6_SIGNAL_FINAL, top=1.28, width=12.2,
          col_widths=(3.4, 1.8, 2.0, 1.5, 3.5), header_size=11.5, body_size=11.5)
    bullets(s, [
        [("Learned DC/energy opacity returns to zero and narrowly improves the held-out score. "
          "Frozen opacity fails; magnitude supervision overfits.", False)],
        [("O4 is the current selected B787 model because NVS is the headline metric.", True)],
    ], top=4.18, size=12.5)

    s = content_slide(prs, "Round 6: complete B787 training and held-out curves")
    picture(s, os.path.join(BR, "r6_error_curves.png"), top=1.32, width=11.2)
    caption(s, "All seven runs complete at epoch 150; markers show each arm's best held-out score.",
            top=6.62, size=12, align_center=True)

    s = content_slide(prs, "Round 6 B787 geometry — fixed threshold t=0.20")
    table(s, R6_GEOM_FIXED_FINAL, top=1.28, width=11.9,
          col_widths=(4.1, 2.1, 1.9, 1.8, 2.0), header_size=11.5, body_size=11.5)
    bullets(s, [
        [("The fixed threshold preserves the signal-quality read: O4/O3 lead the healthy arms, "
          "the magnitude arms lose, and frozen opacity collapses.", True)],
    ], top=4.25, size=12.5)

    s = content_slide(prs, "Round 6 B787 geometry — per-metric oracle thresholds")
    table(s, R6_GEOM_ORACLE_FINAL, top=1.28, width=12.0,
          col_widths=(4.2, 2.0, 1.9, 1.9, 2.0), header_size=11.5, body_size=11.5)
    bullets(s, [
        [("Ground-truth threshold tuning makes the healthy arms look nearly interchangeable "
          "and hides the magnitude-loss NVS failures. Keep this beside the fixed table.", True)],
    ], top=4.25, size=12.5)

    s = content_slide(prs, "Round 6 B787 reconstructions after completion")
    picture(s, os.path.join(BR, "r6_grid.png"), top=1.45, width=12.6)
    caption(s, "Final/best 48³ checkpoints; trilinear upsampling is visualization only. "
               "Cyan = B787 STL truth.", top=6.12, size=12, align_center=True)

    s = content_slide(prs, "B787 radar NVS comparison — RIFT vs SpINR-style and Radar Fields")
    table(s, RADAR_SIGNAL_AUDIT, top=1.45, width=12.2,
          col_widths=(3.3, 2.8, 2.6, 3.5), header_size=11.5, body_size=12)
    bullets(s, [
        [("RIFT beats the isotropic SpINR-style ablation by 3.4812 points, a ", False),
         ("12.20% relative error reduction", True), (".", False)],
        [("Radar Fields is power-only, so it remains N/A in the coherent column; the matched "
          "range-power column now provides the direct signal comparison.", True)],
    ], top=3.35, size=13)

    s = content_slide(prs, "B787 reconstruction at fixed t=0.20 — SpINR leads")
    table(s, RADAR_GEOM_FIXED_AUDIT, top=1.45, width=12.2,
          col_widths=(3.3, 1.9, 1.9, 1.7, 1.7, 1.7), header_size=11, body_size=11.5)
    bullets(s, [
        [("At one common threshold, SpINR is better than RIFT on every listed geometry metric. "
          "Both are far ahead of Radar Fields.", True)],
    ], top=3.35, size=13)

    s = content_slide(prs, "B787 reconstruction after oracle tuning — RIFT leads")
    table(s, RADAR_GEOM_ORACLE_AUDIT, top=1.45, width=12.25,
          col_widths=(3.2, 2.1, 1.8, 1.7, 1.7, 1.75), header_size=10.5, body_size=10.5)
    bullets(s, [
        [("After a separate ground-truth-optimal threshold for every metric, RIFT wins every "
          "column. This is achievable geometry, not a threshold-independent win.", True)],
    ], top=3.35, size=13)

    s = content_slide(prs, "Current best B787 result — what we can claim now")
    bullets(s, [
        [("::Novel-view synthesis", True)],
        [("RIFT reaches ", False), ("25.0540%", True),
         (" global coherent held-out error vs 28.5352% for the SpINR-style isotropic ablation.", False)],
        [("::3D reconstruction", True)],
        [("Fixed threshold favors SpINR; per-metric oracle tuning favors RIFT. The ranking "
          "reversal prevents a threshold-independent geometry claim.", False)],
        [("::Radar Fields", True)],
        [("RIFT is much better geometrically under the common evaluator and now also wins the "
          "matched normalized range-power comparison.", False)],
        [("::Model read", True)],
        [("Learned visibility turns itself off on PEC B787; magnitude supervision overfits. "
          "The degree cap and controlled support remain the reliable improvements.", True)],
    ], top=1.40, size=14)


RIFT_BEST_SIGNAL = [
    ("round", "selected RIFT model", "train coherent ↓", "held-out coherent ↓", "range-power ↓"),
    ("Round 5", "degree 3 + target prune to 20k", "12.5193%", "25.3216%", range_power_pct("r5")),
    ("Round 6", "learned DC opacity · ζ=5.39e−16", "0.9494%", "25.0540%", range_power_pct("r6")),
    ("Round 7", "target 20k + SH-degree prior λ=1e−9", "12.5313%", "25.3372%", range_power_pct("r7")),
]

RIFT_BEST_GEOM_FIXED = [
    ("selected RIFT model · fixed t=.20", "Chamfer m² ↓", "Hausdorff mm ↓", "HD95 mm ↓", "IoU ↑", "F1 ↑"),
    ("Round 5 · target 20k", "1.4159e−4", "51.997", "21.045", "0.1025", "0.5257"),
    ("Round 6 · learned DC", "1.5970e−4", "51.997", "21.641", "0.0878", "0.4688"),
    ("Round 7 · target 20k + λ=1e−9", "1.3846e−4", "48.340", "21.016", "0.1026", "0.5262"),
]

RIFT_BEST_GEOM_ORACLE = [
    ("selected RIFT model · per-metric oracle", "Chamfer ↓", "Hausdorff ↓", "HD95 ↓", "IoU ↑", "F1 ↑"),
    ("Round 5 · target 20k", "4.8642e−5 @.60", "20.489 @.40", "9.781 @.60", "0.2961 @.70", "0.8508 @.60"),
    ("Round 6 · learned DC", "4.8316e−5 @.60", "20.489 @.40", "9.924 @.60", "0.2949 @.70", "0.8560 @.60"),
    ("Round 7 · target 20k + λ=1e−9", "4.8642e−5 @.60", "20.489 @.40", "9.781 @.60", "0.2949 @.70", "0.8508 @.60"),
]

# --------------------------------------------------------------------------
# Baseline suite as audited 2026-08-20. Three baselines landed since the
# 2026-08-09 build: GeRaF (complete, but on its OWN observable), the E28
# matched-filter-backprojection anchor (geometry reference), and RadarSplat S0
# (complete after two failed predecessors; terminal validation is a negative).
# Sources:
#   GeRaF train + eval  scratch1/rift_power_baselines_20260813_v2/
#                       {g0_geraf/checkpoints/history.json,
#                        e0_evaluation/geraf/geraf_geraf_3d/metrics.json}
#   MF-BP anchor        .../a0_e28_anchor/anchor_results.json
#   RadarSplat          .../s0_radarsplat/checkpoints/status.json + Report-12043326.out
# --------------------------------------------------------------------------

B787_METHOD_SIGNAL_SUMMARY = [
    ("method", "status", "train coherent ↓", "validation coherent ↓", "validation range-power ↓"),
    ("RIFT (ours)", "selected joint-output model", "12.5313%", "25.3372%", range_power_pct("r7")),
    ("SpINR-style", "complete", "5.76%", "28.5352%", range_power_pct("spinr")),
    ("Radar Fields", "complete · power-only", "N/A", "N/A", "304.8836%"),
    ("GeRaF", "complete · direct response", geraf_complex_pct("train"),
     geraf_complex_pct("validation"), "N/A"),
    ("Matched-filter BP", "geometry anchor", "N/A", "N/A", "N/A"),
    ("RadarSplat S0", "initial · tuning expected", "N/A", "N/A", "92.7748%*"),
    ("Sugavanam–Ertin", "in progress", "—", "—", "—"),
]

# GeRaF converged but reports on its own 3D matched-filter MAGNITUDE volume
# (per-view 32³, 200/200 held-out views). That is neither our coherent-complex
# score nor our normalized range-power score, and e0_evaluation/ contains a
# geraf/ directory ONLY -- so no RIFT row exists on this observable yet.
B787_GERAF_OBSERVABLE = [
    ("comparison axis", "GeRaF 3-D observable", "coherent complex NVS"),
    ("tensor being compared", "real |matched-filter amplitude| volume", "complex S(f)[Rx,Tx]"),
    ("samples per held-out view", "32 × 32 × 32 voxels", "600 frequencies × 16 × 16 channels"),
    ("phase information", "discarded by magnitude", "preserved"),
    ("pooled relative-MSE formula", "Σ|V̂−V|² / Σ|V|²", "Σ|Ŝ−S|² / Σ|S|²"),
    ("GeRaF result", "27.4973%", "not computed"),
]

B787_GEOM_FIXED_ALL = [
    ("method · fixed t=.20", "Chamfer m² ↓", "Hausdorff mm ↓", "HD95 mm ↓", "IoU ↑", "F1 ↑"),
    ("RIFT (ours)", "1.3846e−4", "48.340", "21.016", "0.1026", "0.5262"),
    ("SpINR-style", "1.2767e−4", "27.436", "20.956", "0.1058", "0.6052"),
    ("Matched-filter BP anchor", "8.0340e−3", "176.148", "159.090", "0.0515", "0.2838"),
    ("Radar Fields", "1.3873e−2", "234.401", "174.765", "0.0009", "0.0062"),
    ("RadarSplat S0 · initial", "N/A", "N/A", "N/A", "N/A", "tuning expected"),
    ("GeRaF", "N/A", "N/A", "N/A", "N/A", "different observable"),
    ("Sugavanam–Ertin", "in progress", "—", "—", "—", "—"),
]

B787_GEOM_ORACLE_ALL = [
    ("method · per-metric oracle", "Chamfer ↓", "Hausdorff ↓", "HD95 ↓", "IoU ↑", "F1 ↑"),
    ("RIFT (ours)", "4.8642e−5 @.60", "20.489 @.40", "9.781 @.60", "0.2949 @.70", "0.8508 @.60"),
    ("SpINR-style", "9.6892e−5 @.40", "23.676 @.25", "15.725 @.40", "0.1111 @.30", "0.6876 @.30"),
    ("Matched-filter BP anchor", "1.1386e−4 @.50", "27.436 @.40", "15.837 @.50", "0.1139 @.30", "0.6551 @.50"),
    ("Radar Fields", "4.3883e−3 @.95", "161.083 @.95", "120.614 @.95", "0.0085 @.95", "0.0546 @.95"),
    ("RadarSplat S0 · initial", "N/A", "N/A", "N/A", "N/A", "tuning expected"),
    ("GeRaF", "N/A", "N/A", "N/A", "N/A", "different observable"),
    ("Sugavanam–Ertin", "in progress", "—", "—", "—", "—"),
]

# --------------------------------------------------------------------------
# Round 8b -- 10k-view B787 re-synthesis. IN PROGRESS (epochs 24-89 of 150 at
# the 2026-08-19 audit; all five arms queued, none running). Source:
#   scratch1/rift_round8b_runs/20260810/*/history.csv
# Same backbone family as R5/R7 (g48, extent 0.15, degree 3, seed 42) but
# 3,200 train / 1,000 val / 1,000 sealed test drawn from the 10k-view npz.
# --------------------------------------------------------------------------

B787_ROUND8B_PARTIAL = [
    ("Round 8b arm (10k-view npz)", "epoch", "train coherent ↓", "best held-out coherent ↓"),
    ("eps 1e−15 · lr 3e−6", "89 / 150", "0.3312%", "0.3785%"),
    ("eps 1e−15 · lr 3e−4", "60 / 150", "0.2994%", "0.3971%"),
    ("legacy · eps 1e−8 · lr 3e−3", "59 / 150", "0.3388%", "0.4754%"),
    ("eps 1e−15 · lr 3e−3", "49 / 150", "0.4140%", "0.5333%"),
    ("eps 1e−15 · lr 3e−5", "24 / 150", "0.5323%", "0.8393%"),
]

# --------------------------------------------------------------------------
# PUBLIC DATASETS -- CVDomes + GOTCHA, partial results as of 2026-08-19.
# Campaign is 18 scenes x 8 native methods = 144 cells; 35 have completed.
# Sources:
#   completion    experiment_state/public_radar/cvdomes_gotcha_v1/
#                 */*/manager_completion.json
#   finite-SH     */finite_sh_interpolation/finite_sh_interpolator.json
#   Radar Fields  */radar_fields/checkpoints/*/radar_fields_history.csv
#   MF-BP focus   */matched_filter_backprojection/matched_filter.npz
# NOTE: nothing here is a cross-method comparison yet -- no scene has a
# complete row, and RIFT itself is 0/18.
# --------------------------------------------------------------------------

PUBLIC_SETUP = [
    ("item", "CVDomes", "GOTCHA"),
    ("nature", "simulated coherent far-field monostatic", "measured unformed coherent SAR"),
    ("scenes", "10 civilian vehicles", "8 independent HH passes"),
    ("support", "10 × 10 m planar · 5 m radius", "100 × 100 m planar · 50 m core"),
    ("phase sign", "−1 (calibrated)", "−1 (8 / 8 passes agree)"),
    ("grid", "64³ over ±2.887 m", "64³ over the core radius"),
    ("split used", "random 1,800 / 200 · seed 42", "random 1,800 / 200 · seed 42"),
    ("split required (§15)", "leakage-safe source-sector", "contiguous azimuth sector"),
]

PUBLIC_COMPLETION = [
    ("native method", "cells complete", "what exists", "signal result yet?"),
    ("RIFT", "0 / 18", "nothing", "NO"),
    ("SpINR-style degree 0", "1 / 18", "one checkpoint", "NO"),
    ("Sugavanam–Ertin", "0 / 18", "nothing", "NO"),
    ("RadarSplat", "0 / 18", "nothing", "NO"),
    ("GeRaF", "3 / 18", "checkpoints only · no eval", "NO"),
    ("Matched-filter backprojection", "9 / 18", "coherent 64³ images", "image only"),
    ("Radar Fields", "10 / 18", "checkpoints + history", "YES"),
    ("Finite-SH interpolation (oracle)", "12 / 18", "degree sweep", "YES"),
    ("TOTAL", "35 / 144", "no scene has a complete row", "no comparison possible"),
]

# The model-free angular interpolator -- which reaches 25.86% on B787 -- selects
# degree 0 on EVERY completed public cell and lands at or above the predict-zero
# floor. This is physics, not a conversion bug (see PUBLIC_MF_FOCUS): CVDomes at
# X-band on a ~5 m vehicle has L = 2kR ~ 1000, i.e. ~1e6 angular DOF against
# 1,800 views, so a degree-32 expansion is nowhere near the band limit.
PUBLIC_FINITE_SH = [
    ("scene", "sel. degree", "held-out ↓", "deg-0 train", "deg-32 train", "deg-32 held-out"),
    ("cvdomes_camry", "0", "101.29%", "99.86%", "89.11%", "8.8e+07 %"),
    ("cvdomes_jeep99", "0", "100.47%", "99.84%", "84.52%", "8.3e+07 %"),
    ("cvdomes_mazdampv", "0", "100.28%", "99.87%", "81.57%", "8.3e+07 %"),
    ("cvdomes_mitsubishi", "0", "100.48%", "99.86%", "82.64%", "9.8e+07 %"),
    ("cvdomes_sentra", "0", "100.70%", "99.93%", "91.52%", "1.2e+08 %"),
    ("cvdomes_toyotaavalon", "0", "100.47%", "99.84%", "83.19%", "2.5e+08 %"),
    ("gotcha_pass2_hh", "0", "100.00%", "100.00%", "99.63%", "1.4e+10 %"),
    ("gotcha_pass3_hh", "0", "100.01%", "100.00%", "102.36%", "7.5e+10 %"),
    ("gotcha_pass4_hh", "0", "100.01%", "100.00%", "153.29%", "5.1e+12 %"),
    ("gotcha_pass5_hh", "0", "100.01%", "100.00%", "101.71%", "5.8e+11 %"),
    ("gotcha_pass6_hh", "0", "100.01%", "100.00%", "101.83%", "8.1e+11 %"),
    ("gotcha_pass7_hh", "0", "100.01%", "100.00%", "99.75%", "1.1e+10 %"),
]

PUBLIC_RADAR_FIELDS = [
    ("scene", "train rel-MSE ↓", "held-out rel-MSE ↓", "held-out PSNR ↑", "moved since step 50?"),
    ("cvdomes_toyotatacoma", "2446.53%", "1597.76%", "9.57 dB", "no (1586% → 1598%)"),
    ("cvdomes_camry", "2075.44%", "1434.42%", "9.61 dB", "no (1444% → 1434%)"),
    ("cvdomes_jeep99", "893.96%", "620.99%", "9.54 dB", "no (637% → 621%)"),
    ("cvdomes_hondacivic4dr", "814.40%", "559.63%", "9.53 dB", "no (586% → 560%)"),
    ("cvdomes_jeep93", "549.79%", "373.29%", "9.46 dB", "no (394% → 373%)"),
    ("gotcha_pass5_hh", "205.99%", "165.84%", "9.81 dB", "no (171% → 166%)"),
    ("gotcha_pass2_hh", "198.37%", "156.68%", "9.58 dB", "no (161% → 157%)"),
    ("gotcha_pass8_hh", "193.73%", "151.97%", "9.99 dB", "no (156% → 152%)"),
    ("gotcha_pass7_hh", "178.84%", "137.83%", "9.93 dB", "no (142% → 138%)"),
    ("gotcha_pass1_hh", "176.35%", "138.42%", "9.48 dB", "no (143% → 138%)"),
]

PUBLIC_MF_FOCUS = [
    ("scene", "peak / median", "dynamic range", "voxels above 1% of peak"),
    ("gotcha_pass1_hh", "29,705", "44.7 dB", "0.17%"),
    ("gotcha_pass3_hh", "15,484", "41.9 dB", "0.39%"),
    ("gotcha_pass4_hh", "14,152", "41.5 dB", "0.44%"),
    ("gotcha_pass2_hh", "7,941", "39.0 dB", "0.81%"),
    ("cvdomes_maxima", "9,660", "39.8 dB", "2.95%"),
    ("cvdomes_hondacivic4dr", "4,453", "36.5 dB", "6.75%"),
    ("cvdomes_mazdampv", "4,492", "36.5 dB", "6.05%"),
    ("cvdomes_jeep93", "2,057", "33.1 dB", "10.35%"),
    ("cvdomes_jeep99", "937", "29.7 dB", "18.57%"),
]

PUBLIC_QUEUE = [
    ("track", "queued jobs", "running", "state"),
    ("PublicRadar (rpd_*)", "48", "0", "35 / 144 cells complete"),
    ("Round 8b · 10k views", "5", "0", "epochs 24–89 of 150"),
    ("Round 8 · 2k sealed split", "5", "0", "epochs 14–30 of 150 · too early to read"),
    ("RadarSplat S0 · B787", "0", "0", "complete · E0 evaluation pending"),
    ("GeRaF G0 · B787", "0", "0", "complete + evaluated"),
]


def add_current_best_b787(prs):
    """Focused deck containing only the selected Round-5 and Round-6 RIFT models."""
    r6_vs_spinr_power = range_power_reduction("r6", "spinr")
    if RANGE_POWER_RESULTS["r6"] is not None:
        r6_vs_radar_fields_power = 100.0 * (
            1.0 - RANGE_POWER_RESULTS["r6"]["normalized_range_power_rel_mse"] / 3.0488364040713822
        )
    else:
        r6_vs_radar_fields_power = None

    s, sub = add_slide(prs, "Title Slide", "RIFT: Current Best Results")
    if sub is not None:
        sub.text_frame.text = ("Daqian Bao — August 20, 2026 · method and results · "
                               "native-complex radar novel-view synthesis and metric 3-D reconstruction")

    s = content_slide(prs, "Method (1/3) — direction-dependent point-scatterer field")
    picture(s, os.path.join(BR, "paper_method_scene.png"), top=1.28, width=11.65)

    s = content_slide(prs, "Method (2/3) — radar signal as Fourier synthesis")
    picture(s, os.path.join(BR, "paper_method_signal.png"), top=1.28, width=11.65)

    s = content_slide(prs, "Method (3/3) — fast evaluation, fitting and sampling")
    picture(s, os.path.join(BR, "paper_method_implementation.png"), top=1.18, width=11.65)

    s = content_slide(prs, "Selected RIFT reconstructions — planform view")
    picture(s, os.path.join(BR, "current_best_planform.png"), top=1.38, width=10.7)

    s = content_slide(prs, "Selected RIFT reconstructions — nose-on view")
    picture(s, os.path.join(BR, "current_best_front.png"), top=1.38, width=10.7)

    s = content_slide(prs, "B787 comparison — status, training and validation signal")
    table(s, B787_METHOD_SIGNAL_SUMMARY, top=1.28, width=12.25,
          col_widths=(2.25, 2.75, 2.35, 2.45, 2.45), header_size=9.75, body_size=10.25)
    bullets(s, [
        [("RIFT appears once: the selected Round-7 joint-output checkpoint supplies both signal "
          "numbers and the geometry reported later.", True)],
        [("GeRaF is benchmarked on its internal complex response before matched filtering: "
          "validation is complete and training is pending the experiment manager. Its prior "
          "27.4973% 3-D magnitude score is intentionally not substituted.", False)],
        [("RadarSplat S0 is an initial result and tuning is expected. The asterisk marks terminal "
          "training-side validation pending formal E0; Sugavanam–Ertin is in progress.", False)],
    ], top=4.10, size=11.75)

    s = content_slide(prs, "B787 NVS lives on a Fibonacci viewing sphere")
    picture(s, os.path.join("figures", "b787_range_power", "fibonacci_sampling.png"),
            top=1.35, width=12.1)
    bullets(s, [
        [("Blue overlays the 1,800 measured training directions; orange marks the 200 held-out "
          "directions interleaved across the same Fibonacci sphere.", False)],
        [("The dense reference uses measured signal at all 2,000 views. On the next slide, held-out "
          "measurements are replaced by model predictions; metrics remain validation-only.", True)],
    ], top=5.55, height=1.35, size=11.5)

    s = content_slide(prs, "B787 NVS — dense composite for selected models and baseline")
    picture(s, os.path.join("figures", "b787_range_power", "nvs_sphere_comparison.png"),
            top=1.08, width=11.75)

    s = content_slide(prs, "B787 learned 3D fields — direct visual baseline comparison")
    picture(s, os.path.join(BR, "baseline_fields_all.png"), top=1.18, width=12.65)
    caption(s, "Visualization only: RadarSplat is binned from its explicit Gaussian centres; "
               "all display contrasts are method-normalized.", top=5.92, size=11,
            align_center=True)

    s = content_slide(prs, "B787 geometry comparison — fixed threshold t=0.20")
    table(s, B787_GEOM_FIXED_ALL, top=1.30, width=12.25,
          col_widths=(2.8, 2.0, 1.9, 1.8, 1.8, 1.95), header_size=10.5, body_size=10.5)
    bullets(s, [
        [("SpINR-style leads the selected RIFT model on every fixed-threshold metric.", True)],
        [("New reference: the ", False), ("matched-filter backprojection anchor", True),
         (". RIFT beats it by 58× on Chamfer, 7.6× on HD95, 3.6× on Hausdorff, "
          "2.0× on IoU and 1.85× on F1.", False)],
        [("RadarSplat S0 is an initial result with tuning expected. GeRaF has no common geometry "
          "row, and Sugavanam–Ertin remains in progress.", False)],
    ], top=4.35, size=11.5)

    s = content_slide(prs, "B787 geometry comparison — per-metric oracle thresholds")
    table(s, B787_GEOM_ORACLE_ALL, top=1.30, width=12.3,
          col_widths=(2.8, 2.1, 1.9, 1.8, 1.85, 1.85), header_size=10, body_size=10)
    bullets(s, [
        [("After ground-truth threshold tuning, the selected RIFT model leads every reported "
          "baseline on all five geometry metrics.", True)],
        [("The anchor's oracle optimum sits at n_points = 112, right on the degenerate-level-set "
          "floor — quote the fixed-threshold column for it, not this one.", True)],
        [("This ranking is achievable geometry after calibration, not a threshold-independent win.", False)],
    ], top=4.35, size=11.5)

    s = content_slide(prs, "Round 8b — 10k-view B787, IN PROGRESS and not yet quotable")
    table(s, B787_ROUND8B_PARTIAL, top=1.40, width=11.6, left=0.85,
          col_widths=(4.1, 2.1, 2.6, 2.8), header_size=11.5, body_size=11.5)
    bullets(s, [
        [("::WORK IN PROGRESS — five arms at epochs 24–89 of 150, all queued, none running.", True)],
        [("Same backbone family as Rounds 5/7 (48³, ±0.15 m, degree 3, seed 42), but 3,200 train "
          "/ 1,000 val / 1,000 sealed test from the 10k-view re-synthesis.", False)],
        [("Held-out coherent error reads ~0.4% against the 25% floor — a 60× collapse from "
          "1.78× more training views.", True)],
        [("Our own measured view-count slope (0.65 points per 100 views at n=1800, decelerating) "
          "predicts ~16%, not 0.4%. ", False),
         ("Do not quote this until the model-free oracle is re-run on the exact R8b split.", True)],
        [("Three candidates the artifacts cannot separate: 1,000 val views sit √5 ≈ 2.24× closer "
          "in angle; the split may leak; the re-synthesis may differ beyond view count.", False)],
    ], top=3.45, size=11.5)

    add_public_radar_partial(prs)

    s = content_slide(prs, "Current best result — concise read")
    bullets(s, [
        [("::B787 — selected model", True)],
        [("Round 7 (target 20k + SH-degree prior λ=1e−9) is the single joint-output benchmark: "
          "held-out coherent 25.3372% and the reported RIFT geometry row all come from this one "
          "checkpoint.", False)],
        [("::B787 — baselines", True)],
        [("RIFT beats SpINR-style on coherent NVS and beats the matched-filter anchor on all "
          "geometry. SpINR still wins fixed-threshold geometry; RIFT wins after oracle tuning. "
          "GeRaF validation is now evaluated on the raw complex response while its training score "
          "is pending the experiment manager; RadarSplat S0 is an initial result with tuning "
          "expected; Sugavanam–Ertin is in progress.", False)],
        [("::Public data — what we actually have", True)],
        [("35 of 144 cells. ", False), ("RIFT is 0 / 18", True),
         (", so no cross-method comparison exists on CVDomes or GOTCHA yet — no scene has a "
          "complete row.", False)],
        [("::Public data — the one real finding", True)],
        [("The model-free angular interpolator collapses to the predict-zero floor on every "
          "completed cell. On B787 it reaches 25.86% and RIFT leads it by only 0.8 point; here "
          "any scene-free predictor is pinned at 100%, so the headroom is far larger — and "
          "entirely unclaimed.", True)],
        [("::Next three actions", True)],
        [("(1) run the model-free oracle on the R8b 10k split — minutes of CPU, decides whether "
          "0.4% is a headline or an artifact; (2) finish GeRaF's training complex-response score and "
          "RadarSplat tuning/E0; "
          "(3) settle the public-data split contract before more of the 144 cells burn.", False)],
    ], top=1.30, height=5.9, size=11.5)


def add_public_radar_partial(prs):
    """CVDomes + GOTCHA partial results, audited 2026-08-19.

    Everything on these slides is explicitly PARTIAL. The campaign is 18 scenes
    x 8 native methods; 35 cells have completed and RIFT is 0/18, so there is
    no cross-method comparison to show yet. What IS solid is (a) the data
    converts and focuses coherently, and (b) the model-free interpolation
    bound that constrains the whole B787 story does not exist here.
    """
    s = content_slide(prs, "Public datasets — CVDomes and GOTCHA contract")
    table(s, PUBLIC_SETUP, top=1.40, width=12.2, left=0.55,
          col_widths=(2.9, 4.3, 5.0), header_size=11.5, body_size=11)
    bullets(s, [
        [("Neither dataset has our 16 Tx × 16 Rx bistatic MIMO contract, so both run through a "
          "monostatic specialization.", False)],
        [("::PROTOCOL DEVIATION — the completed cells used a RANDOM split, not the leakage-safe "
          "sector split the plan requires. Radar Fields also validated on 2,304 views, not 200, "
          "so the two completed method families are not even on the same split as each other.", True)],
    ], top=5.05, size=11.5)

    s = content_slide(prs, "Public datasets — what we currently have: 35 of 144 cells")
    table(s, PUBLIC_COMPLETION, top=1.40, width=11.5, left=0.9,
          col_widths=(4.2, 2.3, 3.2, 1.8), header_size=11.5, body_size=11)
    bullets(s, [
        [("::No scene has a complete row, so zero cross-method comparisons are possible today.", True)],
        [("RIFT, Sugavanam–Ertin and RadarSplat are all at 0 / 18. What has completed is dominated "
          "by the cheap methods: the interpolation oracle, matched-filter backprojection and "
          "Radar Fields.", False)],
        [("48 PublicRadar jobs are queued and none is running — the campaign is throughput-blocked "
          "on preemption, not blocked on code.", False)],
    ], top=4.85, size=11.5)

    s = content_slide(prs, "Public datasets — the model-free interpolation bound does NOT exist here")
    table(s, PUBLIC_FINITE_SH, top=1.12, width=12.3, left=0.5,
          col_widths=(3.1, 1.5, 1.9, 1.9, 1.9, 2.0), header_size=10, body_size=9.5)
    bullets(s, [
        [("Every completed cell selects ", False), ("degree 0", True),
         (" and lands at or above the predict-zero floor — on B787 the same oracle reaches "
          "25.86%. Physics, not a bug: CVDomes at X-band on a ~5 m vehicle has L = 2kR ≈ 1000, "
          "i.e. ~10⁶ angular DOF against 1,800 views.", True)],
        [("::CAVEAT — gotcha_pass4's deg-32 TRAIN error (153.29%) exceeds its own deg-0 train "
          "error (100.00%). A least-squares fit cannot do that; that cell is a numerical failure.", True)],
    ], top=5.42, height=1.9, size=10.5)

    s = content_slide(prs, "Public datasets — Radar Fields is at the trivial floor everywhere")
    table(s, PUBLIC_RADAR_FIELDS, top=1.32, width=12.0, left=0.65,
          col_widths=(3.3, 2.3, 2.4, 2.0, 2.0), header_size=11, body_size=10.5)
    bullets(s, [
        [("Held-out error is 138–1598% on every scene, and it is essentially ", False),
         ("flat from step 50 to step 800", True),
         (" — it cannot fit even its own training views.", False)],
        [("Held-out RMSE sits at 0.32–0.34 and PSNR at 9.3–10.0 dB regardless of scene: the "
          "signature of an uncorrelated predictor.", False)],
        [("This is NOT public-data-specific — Radar Fields reads 304.8836% on B787 at the same "
          "800-step budget. Decide whether 800 steps is the intended budget before this enters "
          "any table; as it stands the row says nothing about the datasets.", True)],
    ], top=4.85, size=11.5)

    s = content_slide(prs, "Public datasets — the conversion is sound: coherent focus confirmed")
    table(s, PUBLIC_MF_FOCUS, top=1.38, width=11.0, left=1.15,
          col_widths=(3.6, 2.4, 2.4, 2.6), header_size=11.5, body_size=11.5)
    bullets(s, [
        [("Matched-filter backprojection focuses properly on every completed scene: 30–45 dB "
          "peak-to-median, with GOTCHA support tight at 0.17–0.81% of voxels above 1% of peak.", True)],
        [("Forward phase sign −1 was determined empirically, with all eight GOTCHA HH passes "
          "selecting it independently and target-localization offsets ≤ 0.722 m.", False)],
        [("So the geometry and phase conventions are right, and the interpolator's failure above "
          "is a statement about angular sampling — not about the data pipeline.", True)],
    ], top=4.75, size=11.5)

    s = content_slide(prs, "Work in progress — nothing is running, everything is queued")
    table(s, PUBLIC_QUEUE, top=1.45, width=11.2, left=1.05,
          col_widths=(3.5, 2.2, 1.8, 3.7), header_size=12, body_size=12)
    bullets(s, [
        [("::All figures on the public-data slides are PARTIAL and will move.", True)],
        [("Round 8/8b arms were resubmitted 2026-08-18 under a new local submission manager after "
          "a numpy import failure in the batch environment was fixed at the conda env.", False)],
        [("Standalone B787 RadarSplat S0 is complete; only its formal E0 evaluation remains. "
          "The separate PublicRadar RadarSplat lane is still 0 / 18 and queued with the campaign.", True)],
    ], top=3.55, size=12)


def drop_all_slides(prs):
    """Strip the base deck's slides but keep its GT template/layouts.

    The 2026-07-13 sphere deck is still the template source; from 2026-08-06
    its 10 slides are no longer carried verbatim, because Part 1 compresses
    that round to its tables. The originals stay on disk in
    slides/RIFT_PEC_Sphere_bw3ghz_Results_2026-07-13.pptx.
    """
    id_list = prs.slides._sldIdLst
    rel_key = ("{http://schemas.openxmlformats.org/officeDocument/2006/"
               "relationships}id")
    for sld in list(id_list):
        prs.part.drop_rel(sld.get(rel_key))
        id_list.remove(sld)


def add_deck_title(prs):
    s, sub = add_slide(prs, "Title Slide", "RIFT: Results and Audited Comparisons")
    if sub is not None:
        sub.text_frame.text = ("Daqian Bao — August 20, 2026 · historical results · storyline · "
                               "completed Round 6 · audited RadarSplat S0 result")


def main():
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    template = next((p for p in (BASE, OUT, OUT_BEST) if os.path.exists(p)), None)
    if template is None:
        raise FileNotFoundError(
            f"No deck template found; expected one of {BASE}, {OUT}, or {OUT_BEST}"
        )

    series = Presentation(template)
    drop_all_slides(series)
    add_deck_title(series)
    add_part1_initial_results(series)
    add_competitive_strategy(series)
    add_literature_review(series)
    add_part3_round6(series)
    add_part4_audited_comparisons(series)
    series.save(OUT)

    current_best = Presentation(template)
    drop_all_slides(current_best)
    add_current_best_b787(current_best)
    current_best.save(OUT_BEST)

    print(f"Wrote {OUT}: {len(series.slides)} slides")
    print(f"Wrote {OUT_BEST}: {len(current_best.slides)} slides")


if __name__ == "__main__":
    main()
