#!/usr/bin/env python3
"""Problem-illustration schematic of RIFT: one fitted field answers the forward and the inverse problem.

Notation follows the paper (writing agent, 2026-09-25): measured S_v and rendered S-hat_v (bold view-level arrays,
Eq. forward), training viewpoints V_tr, held-out calibrated viewpoint v*, the data term of Eq. objective (regularizers
omitted), "learned point-scattering field" (no collective symbol), "maximum-intensity projection" spelled out, and the
introduction's forward-inverse pair (novel-view synthesis: scene -> measurements; reconstruction: measurements ->
scene).

(a) The scene, the B787 stand-in mesh (the paper's plan-view render), inside its sphere of radar viewpoints: TRAIN
    directions (dots; true directions seen by the same orthographic camera, sphere not to scale), a drawn parabolic
    radar, and a path of viewpoints (the elevation ring of ``prepare_rift_schematic_data_pvc.py``) through v*.
(b) RIFT: the field to learn, the B787's learned point scatterers (plan view) with the horizontal-plane magnitude
    of the learned direction-dependent response |rho_p(u)| of a few points drawn as lobes, and the data term.
(c) Forward problem: the complex response at the band centre, S_v(f_c) (real part solid, imaginary part dotted),
    along the path, RIFT rendered at every viewpoint of the ring (including views whose responses are withheld) against the
    measured TRAIN and held-out VALIDATION views; inverse problem: the
    reference mesh and the learned field's maximum-intensity projection from the same plan camera and crop.

Palette and type are the paper's: RIFT red-orange (inferno 0.60), measurements and structure in ink and gray,
fields on the inferno scale of the view figures, text in Times (TeX Gyre Termes; calligraphic letters from cmsy10).
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
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Arc, Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts_pvc.paper_figure_font import retarget_svg_family, use_paper_font  # noqa: E402

ACCENT = "#dd513a"        # RIFT: inferno(0.60)
ACCENT_TINT = "#fdf1ee"
INK, MUTED, BASE, GRID, LIGHT = "#0b0b0b", "#52514e", "#6b6a66", "#c9c8c3", "#e4e3df"
PANELS = Path("manuscripts/iclr27/figures/rift_dataset/b787")
DATA = Path("/scratch/user/u.db364833/RIFT_runs/paper_figure_drafts_20260924/schematic_draft/schematic_data.npz")
SPHERE = Path("/scratch/user/u.db364833/RIFT_runs/sphere_response_20260924/b787/measured.npz")
# one hue at rising saturation: a point's |rho_p(u)| from weak (pale) to its peak (RIFT red-orange)
SATURATION = LinearSegmentedColormap.from_list("rift_saturation", ["#fcefeb", "#f3b9aa", "#e8836c", "#dd513a"])
WINDOW = (-70.0, 20.0)    # azimuth window of the path (deg, nose at 0): around v*, clear of the broadside flashes


def crop_box(image, level=0.08, pad=0.06):
    lum = image[..., :3].mean(-1)
    ys, xs = np.nonzero(lum > level)
    h, w = lum.shape
    py, px = int(pad * h), int(pad * w)
    return max(ys.min() - py, 0), min(ys.max() + py, h), max(xs.min() - px, 0), min(xs.max() + px, w)


def mesh_on_white(path):
    """The plan-view mesh render with its dark background made transparent (the aircraft keeps its shading)."""
    image = mpimg.imread(path)[..., :3][4:-4, 4:-4]          # drop the panel's frame
    y0, y1, x0, x1 = crop_box(image, level=0.2)
    image = image[y0:y1, x0:x1]
    return np.dstack([image, np.clip((image.mean(-1) - 0.12) / 0.12, 0, 1)])


def draw_dish(ax, centre, size, pointing_deg, zorder=6):
    """A stereotypical parabolic radar: a thick reflector bowl on a pivot and post, a feed arm with struts and a
    feed horn at the focus, and wavefronts; the bowl opens along ``pointing_deg`` (page degrees)."""
    c, s = np.cos(np.radians(pointing_deg)), np.sin(np.radians(pointing_deg))

    def place(points):
        points = np.asarray(points, dtype=float) * size
        return np.stack([points[:, 0] * c - points[:, 1] * s, points[:, 0] * s + points[:, 1] * c], 1) + centre

    y = np.linspace(-0.8, 0.8, 41)
    bowl = np.stack([0.55 * (y / 0.8) ** 2 - 0.55, y], 1)          # vertex (-0.55, 0), rim (0, +-0.8), opens +x
    focus = np.array([0.32, 0.0])
    ax.add_patch(Polygon(place(bowl), closed=True, facecolor=LIGHT, edgecolor="none", zorder=zorder))
    ax.plot(*place(bowl).T, color=INK, linewidth=1.3, solid_capstyle="round", zorder=zorder + 1)
    for rim in (bowl[0], bowl[-1]):
        ax.plot(*place([rim, focus]).T, color=INK, linewidth=0.45, zorder=zorder + 1)
    ax.plot(*place([[-0.55, 0.0], focus]).T, color=INK, linewidth=0.7, zorder=zorder + 1)
    ax.add_patch(Circle(place([focus + [0.05, 0.0]])[0], 0.07 * size, facecolor=INK, edgecolor=INK,
                        zorder=zorder + 2))
    placed_bowl = place(bowl)                                          # the post hangs from the bowl's lowest point,
    pivot = placed_bowl[np.argmin(placed_bowl[:, 1])] + np.array([0.0, -0.04 * size])   # so it shows at any aim
    foot = pivot + np.array([0.0, -0.60 * size])                     # post in the page's vertical
    ax.plot([pivot[0], foot[0]], [pivot[1], foot[1]], color=INK, linewidth=1.1, solid_capstyle="butt",
            zorder=zorder - 1)
    ax.add_patch(Circle(pivot, 0.06 * size, facecolor=INK, edgecolor=INK, zorder=zorder))
    ax.add_patch(Polygon([foot + [-0.30 * size, -0.07 * size], foot + [0.30 * size, -0.07 * size],
                          foot + [0.16 * size, 0.05 * size], foot + [-0.16 * size, 0.05 * size]],
                         closed=True, facecolor=INK, edgecolor=INK, linewidth=0.3, zorder=zorder - 1))
    for k, radius in enumerate((0.30, 0.50, 0.70)):                  # wavefronts toward the scene
        ax.add_patch(Arc(place([focus + [0.12, 0.0]])[0], 2 * radius * size, 2 * radius * size,
                         angle=pointing_deg, theta1=-32, theta2=32, color=MUTED, linewidth=0.6,
                         alpha=0.95 - 0.25 * k, zorder=zorder))


def sh_sphere(coefficients, basis_degree, view, npx=140):
    """A point's learned response |rho_p(u)| on the sphere of directions u, seen by the camera ``view`` = (right, up,
    towards-camera) unit vectors of panel (b): RGBA image of the visible hemisphere on [-1, 1]^2 (saturation of one
    hue, from the point's minimum to its maximum over the whole sphere)."""
    import torch
    from rift.spherical_harmonics import real_sh_basis
    from scripts.eval_b787_range_power import theta_phi
    degree = int(round(np.sqrt(len(basis_degree)))) - 1

    def magnitude(directions):
        theta = np.array([theta_phi(d)[0] for d in directions])
        phi = np.array([theta_phi(d)[1] for d in directions])
        basis = real_sh_basis(torch.as_tensor(theta), torch.as_tensor(phi), degree).numpy()   # [basis, N]
        return np.abs(coefficients @ basis)

    right, up, cam = view
    grid = np.linspace(-1, 1, npx)
    a, b = np.meshgrid(grid, -grid)
    inside = a ** 2 + b ** 2 <= 1.0
    depth = np.sqrt(np.clip(1.0 - a ** 2 - b ** 2, 0, None))
    points = (a[..., None] * right + b[..., None] * up + depth[..., None] * cam)[inside]
    golden = np.pi * (3.0 - np.sqrt(5.0))                              # the whole sphere, for the peak
    k = np.arange(2000) + 0.5
    zz = 1 - 2 * k / 2000
    everywhere = np.stack([np.sqrt(1 - zz ** 2) * np.cos(golden * k), np.sqrt(1 - zz ** 2) * np.sin(golden * k), zz], 1)
    whole = magnitude(everywhere)
    low, high = whole.min(), max(whole.max(), 1e-30)
    rgba = np.zeros((npx, npx, 4))
    rgba[inside] = SATURATION(np.clip((magnitude(points) - low) / (high - low + 1e-30), 0, 1))
    return rgba


def sh_structure(coefficients, basis_degree, n=2000):
    """How much each point's |rho_p(u)| varies over the sphere of directions: std / mean over a Fibonacci sphere."""
    import torch
    from rift.spherical_harmonics import real_sh_basis
    degree = int(round(np.sqrt(len(basis_degree)))) - 1
    k = np.arange(n) + 0.5
    z = 1 - 2 * k / n
    golden = np.pi * (3.0 - np.sqrt(5.0))
    theta, phi = np.arccos(z), (golden * k) % (2 * np.pi)
    basis = real_sh_basis(torch.as_tensor(theta), torch.as_tensor(phi), degree).numpy()
    magnitude = np.abs(coefficients @ basis)
    return magnitude.std(1) / np.maximum(magnitude.mean(1), 1e-30)


def camera(azimuth_deg, elevation_deg):
    """Orthographic camera in the B787 frame (nose +z, up +y, left +x): azimuth from the nose toward the left
    wing, elevation above the horizontal. Returns (right, up, towards-camera) unit vectors."""
    az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
    nose, world_up, left = np.array([0.0, 0, 1]), np.array([0.0, 1, 0]), np.array([1.0, 0, 0])
    towards = np.cos(el) * (np.cos(az) * nose + np.sin(az) * left) + np.sin(el) * world_up
    right = np.cross(-towards, world_up)
    right /= np.linalg.norm(right)
    return right, np.cross(right, -towards), towards


def draw_lattice_box(ax, half, pitch, view, zorder=1):
    """The field's 3D grid: a cube of half-width ``half`` about the origin, grid lines every ``pitch`` on its three
    far faces (as the panes of a 3D plot), near edges solid and hidden edges dashed."""
    right, up, towards = view

    def proj(pts):
        pts = np.asarray(pts)
        return pts @ right, pts @ up

    ticks = np.arange(-half, half + 1e-9, pitch)
    far = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            normal = np.zeros(3)
            normal[axis] = sign
            if normal @ towards < 0:                                  # faces away from the camera: draw its grid
                far.append((axis, sign))
                a, b = [i for i in range(3) if i != axis]
                for t in ticks:
                    for vary, fixed in ((a, b), (b, a)):
                        pts = np.zeros((2, 3))
                        pts[:, axis] = sign * half
                        pts[:, fixed] = t
                        pts[:, vary] = [-half, half]
                        ax.plot(*proj(pts), color=GRID, linewidth=0.35, zorder=zorder)
    for axis in range(3):                                             # the 12 edges
        others = [i for i in range(3) if i != axis]
        for s1 in (-1.0, 1.0):
            for s2 in (-1.0, 1.0):
                pts = np.zeros((2, 3))
                pts[:, axis] = [-half, half]
                pts[:, others[0]], pts[:, others[1]] = s1 * half, s2 * half
                hidden = (others[0], s1) in far and (others[1], s2) in far
                ax.plot(*proj(pts), color=MUTED if hidden else INK, linewidth=0.4 if hidden else 0.55,
                        linestyle=(0, (2, 1.5)) if hidden else "-", zorder=zorder if hidden else zorder + 3)


def draw_sh_sphere(ax, centre, radius, rgba, basis, zorder=4):
    """Place a sh_sphere image with its outline, equator and the nose meridian (visible parts)."""
    right, up, cam = basis
    gx, gy = centre
    ax.imshow(rgba, extent=(gx - radius, gx + radius, gy - radius, gy + radius), interpolation="bilinear",
              zorder=zorder)
    t = np.linspace(0, 2 * np.pi, 200)
    for curve in (np.cos(t)[:, None] * np.array([1.0, 0, 0]) + np.sin(t)[:, None] * np.array([0, 0, 1.0]),
                  np.cos(t)[:, None] * np.array([0, 1.0, 0]) + np.sin(t)[:, None] * np.array([0, 0, 1.0])):
        visible = curve @ cam > 0
        xs = np.where(visible, gx + radius * (curve @ right), np.nan)
        ys = np.where(visible, gy + radius * (curve @ up), np.nan)
        ax.plot(xs, ys, color=INK, alpha=0.35, linewidth=0.35, zorder=zorder + 1)
    ax.add_patch(Circle((gx, gy), radius, fill=False, edgecolor=INK, linewidth=0.5, zorder=zorder + 1))


def arrow(fig, start, end, text=None, text_xy=None, color=INK, rad=0.0, ha="center"):
    fig.patches.append(FancyArrowPatch(start, end, transform=fig.dpi_scale_trans, arrowstyle="-|>",
                                       mutation_scale=7, linewidth=0.8, color=color,
                                       connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))
    if text:
        fig.text(*text_xy, text, transform=fig.dpi_scale_trans, ha=ha, va="center", fontsize=6.8, color=color)


def tile(ax, image, box):
    y0, y1, x0, x1 = box
    ax.imshow(image[y0:y1, x0:x1], interpolation="bilinear")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#252525")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--formats", default="pdf,svg,png")
    args = p.parse_args()
    use_paper_font()
    plt.rcParams["mathtext.cal"] = "cmsy10"    # the paper's \mathcal (Termes has no calligraphic letters)
    plt.rcParams.update({"font.size": 7, "axes.edgecolor": MUTED, "axes.linewidth": 0.5,
                         "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 6, "ytick.labelsize": 6})
    data = np.load(DATA)
    meta = json.loads(str(data["meta"]))

    W, H = 5.5, 2.30          # the placeholder's height in the manuscript (page budget)
    fig = plt.figure(figsize=(W, H))

    def axes_in(x0, y0, w, h, **kw):
        return fig.add_axes([x0 / W, y0 / H, w / W, h / H], **kw)

    def label(x, y, text, **kw):
        kw = dict(dict(ha="left", va="top", fontsize=7, color=INK), **kw)
        fig.text(x / W, y / H, text, **kw)

    top = H - 0.04
    label(0.05, top, "(a) scene and radar viewpoints", fontsize=7.5, fontweight="bold")
    label(1.86, top, "(b) the field to learn", fontsize=7.5, fontweight="bold")
    label(3.60, top, "(c) one fitted field, two problems", fontsize=7.5, fontweight="bold")

    # ---- (a) scene, viewpoints, radar, path ------------------------------------------------------------------
    with np.load(SPHERE) as z:
        positions, roles = z["viewpoint_positions"], z["roles"]
    unit = positions / np.linalg.norm(positions, axis=1, keepdims=True)
    visible = (unit[:, 1] > 0) & (roles == "train")                 # plan camera on +y: right = +z, up = +x
    sx0, sy0, ss = 0.10, 0.36, 1.56
    ax = axes_in(sx0, sy0, ss, ss)
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Circle((0, 0), 1.0, fill=False, edgecolor=GRID, linewidth=0.6))
    ax.add_patch(Ellipse((0, 0), 2.0, 0.55, fill=False, edgecolor=GRID, linewidth=0.4, linestyle=(0, (2, 2))))
    shade = 0.35 + 0.65 * unit[visible, 1]
    ax.scatter(unit[visible, 2], unit[visible, 0], s=0.6, c=[(0.42, 0.416, 0.40, a) for a in shade * 0.8],
               linewidths=0, zorder=2)
    ring = np.cos(np.radians(meta["elevation_deg"]))
    star = data["ring_unit"][data["ring_views"] == data["v_star"]][0]
    radar_angle = 130.0                                               # the radar's bearing from the centre (deg)
    # display only: the path and v* are rotated about the sphere's centre so v* lies on the radar's line of sight
    spin = np.radians(radar_angle) - np.arctan2(star[0], star[2])
    arc = np.radians(np.linspace(*WINDOW, 60)) + spin                 # plan view: nose (+z) right, left (+x) up
    ax.plot(ring * np.cos(arc), ring * np.sin(arc), color=INK, linewidth=0.9, linestyle=(0, (3, 1.5)), zorder=3)
    mesh = mesh_on_white(PANELS / "b787_mesh_view3.png")
    half = 0.50
    ax.imshow(mesh, extent=(-half, half, -half * mesh.shape[0] / mesh.shape[1], half * mesh.shape[0] / mesh.shape[1]),
              zorder=4)
    star_xy = ring * np.array([np.cos(np.radians(radar_angle)), np.sin(np.radians(radar_angle))])
    ax.add_patch(Circle(star_xy, 0.06, facecolor="white", edgecolor=INK, linewidth=0.9, zorder=5))
    ax.text(star_xy[0] + 0.02, star_xy[1] - 0.09, "$v^\\star$", ha="left", va="top", fontsize=7.5, color=INK, zorder=5)
    mid = np.radians(-25.0) + spin                                    # outside the arc
    ax.text(1.10 * ring * np.cos(mid), 1.10 * ring * np.sin(mid), "path", va="center", fontsize=6.3,
            ha="left" if np.cos(mid) >= 0 else "right",                # extends away from the arc
            color=INK)
    # the radar: its own corner axes, a dashed beam to the scene
    scene_centre = np.array([sx0 + ss / 2, sy0 + ss / 2])
    rim = ss / 2 / 1.08                                               # the unit sphere's radius in inches
    # outside the sphere, so the wavefronts clear its outline
    dish_centre = scene_centre + (rim + 0.24) * np.array([np.cos(np.radians(radar_angle)),
                                                          np.sin(np.radians(radar_angle))])
    da = axes_in(dish_centre[0] - 0.26, dish_centre[1] - 0.26, 0.52, 0.52)
    da.set_xlim(-1, 1)
    da.set_ylim(-1, 1)
    da.set_aspect("equal")
    da.axis("off")
    heading = np.degrees(np.arctan2(*(scene_centre - dish_centre)[::-1]))
    draw_dish(da, np.array([0.0, 0.10]), 0.62, heading)
    towards = (scene_centre - dish_centre) / np.linalg.norm(scene_centre - dish_centre)
    beam_from, beam_to = dish_centre + 0.20 * towards, scene_centre - 0.24 * towards
    fig.patches.append(FancyArrowPatch(beam_from, beam_to, transform=fig.dpi_scale_trans, arrowstyle="-",
                                       linewidth=0.6, linestyle=(0, (3, 2)), color=MUTED))
    label(sx0 + ss / 2, 0.32, "measured responses $\\mathbf{S}_v$ at\ntraining viewpoints $v \\in \\mathcal{V}_{\\mathrm{tr}}$",
          ha="center", linespacing=1.1)

    # ---- (b) the field to learn --------------------------------------------------------------------------------
    bx0, by0, bw, bh = 1.86, 0.24, 1.56, 1.80
    fig.patches.append(FancyBboxPatch((bx0, by0), bw, bh, transform=fig.dpi_scale_trans, zorder=-1,
                                      boxstyle="round,pad=0.02,rounding_size=0.08", facecolor=ACCENT_TINT,
                                      edgecolor=ACCENT, linewidth=0.9))
    cx = bx0 + bw / 2
    label(cx, by0 + bh - 0.06, "RIFT", ha="center", fontsize=8, fontweight="bold")
    fx = axes_in(bx0 + 0.04, by0 + 0.66, bw - 0.08, 0.92)
    fx.patch.set_alpha(0)
    view = camera(30.0, 65.0)                                          # front-left, steeply from above
    right, up, towards = view
    xyz, energy, orders = data["point_xyz"].astype(np.float64), data["point_energy"], data["point_order"]
    half = 0.055                                                       # the grid about the aircraft (m)
    draw_lattice_box(fx, half, half / 2, view)
    rng = np.random.default_rng(0)
    keep = rng.choice(len(xyz), size=min(6000, len(xyz)), replace=False, p=energy / energy.sum())
    keep = keep[np.all(np.abs(xyz[keep]) <= half, axis=1)]
    keep = keep[np.argsort(xyz[keep] @ towards)]                       # far first, near on top
    depth = xyz[keep] @ towards
    near = (depth - depth.min()) / (depth.max() - depth.min() + 1e-30)
    fx.scatter(xyz[keep] @ right, xyz[keep] @ up, s=0.25 + 0.9 * np.sqrt(energy[keep] / energy.max()),
               color=[(0.867, 0.318, 0.227, 0.3 + 0.6 * a) for a in near], linewidths=0, zorder=2)
    # spherical-harmonic callouts: strong on-body points with the most energy in degrees >= 2, seen by the same camera
    cand = data["lobe_index"]
    cxyz, ce, co = xyz[cand], energy[cand], orders[cand]
    body = np.all(np.abs(cxyz) <= np.array([0.047, 0.022, 0.047]), axis=1)
    picks = {"nose": body & (co >= 1) & (cxyz[:, 2] > 0.028), "wing": body & (np.abs(cxyz[:, 0]) > 0.024),
             "tail": body & (co >= 1) & (cxyz[:, 2] < -0.028)}
    band_energy = np.abs(data["lobe_coefficients"]) ** 2
    degree_of = data["lobe_basis_degree"]
    high = band_energy[:, degree_of >= 2].sum(1) / np.maximum(band_energy.sum(1), 1e-30)
    chosen = {}
    for name, mask in picks.items():
        strong = mask & (ce >= np.median(ce[mask])) if mask.any() else mask
        if strong.any():
            chosen[name] = int(np.flatnonzero(strong)[np.argmax(high[strong])])
    corners = np.array([[x * right + y * up + z * towards for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]])
    box_x = np.array([np.array([a, b, c]) * half for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]) @ right
    box_y = np.array([np.array([a, b, c]) * half for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]) @ up
    radius = 0.37 * half
    slots = {"left-upper": (box_x.min() - 1.35 * radius, 0.45 * box_y.max()),
             "left-lower": (box_x.min() - 1.35 * radius, 0.55 * box_y.min()),
             "right-upper": (box_x.max() + 1.35 * radius, 0.45 * box_y.max()),
             "right-lower": (box_x.max() + 1.35 * radius, 0.55 * box_y.min())}
    free = dict(slots)
    for name, k in sorted(chosen.items(), key=lambda item: -abs(xyz[cand[item[1]]] @ right)):
        index = cand[k]
        px, py = xyz[index] @ right, xyz[index] @ up
        slot = min(free, key=lambda n: np.hypot(free[n][0] - px, free[n][1] - py))
        gx, gy = free.pop(slot)
        rgba = sh_sphere(data["lobe_coefficients"][k], data["lobe_basis_degree"], view)
        towards2d = np.array([px - gx, py - gy]) / np.hypot(px - gx, py - gy)
        fx.plot([px, gx + radius * towards2d[0]], [py, gy + radius * towards2d[1]], color=MUTED, linewidth=0.4,
                zorder=5)
        draw_sh_sphere(fx, (gx, gy), radius, rgba, view, zorder=6)
        fx.scatter([px], [py], s=7, color=ACCENT, edgecolors=INK, linewidths=0.3, zorder=8)
        fx.text(gx, gy - radius - 0.06 * half, f"$L_p={int(orders[index])}$", ha="center", va="top", fontsize=5.6,
                color=INK, zorder=8)
    xs = np.concatenate([box_x, [v[0] - radius for v in slots.values()], [v[0] + radius for v in slots.values()]])
    ys = np.concatenate([box_y, [v[1] - radius * 1.5 for v in slots.values()]])
    fx.set_xlim(xs.min() - 0.02 * half, xs.max() + 0.02 * half)
    fx.set_ylim(ys.min() - 0.02 * half, ys.max() + 0.02 * half)
    fx.set_aspect("equal")
    fx.axis("off")
    label(cx, by0 + 0.64, "learned point scatterers $\\mathbf{x}_p$, each with a\n"
          "spherical-harmonic response $\\rho_p(\\mathbf{u})$; gain $g$", ha="center", fontsize=6.4, color=INK,
          linespacing=1.15)
    label(cx, by0 + 0.24, "$\\min\\;\\sum_{v \\in \\mathcal{V}_{\\mathrm{tr}}} \\ell_v(\\widehat{\\mathbf{S}}_v,\\,"
          " \\mathbf{S}_v)$", ha="center", va="center", fontsize=7.5)
    arrow(fig, (1.63, 1.14), (bx0 - 0.05, 1.14), "fit to\n$\\mathbf{S}_v$", (1.69, 1.36))

    # ---- (c) forward: power along the path; inverse: mesh and learned field ----------------------------------
    label(3.66, 2.09, "forward problem: complex radar NVS", va="center")
    arrow(fig, (bx0 + bw + 0.03, 1.72), (3.61, 2.07), color=ACCENT, rad=-0.2)
    wrapped = (data["ring_azimuth_deg"] + 180.0) % 360.0 - 180.0
    inside = (wrapped >= WINDOW[0]) & (wrapped <= WINDOW[1])
    order_ = np.argsort(wrapped[inside])
    pick = np.flatnonzero(inside)[order_]
    az, ring_roles, views = wrapped[pick], data["ring_roles"][pick], data["ring_views"][pick]
    centre = int(data["centre_index"])
    rift_c = data["rift_spectra"][pick, centre].astype(np.complex128)
    meas_c = data["measured_spectra"][pick, centre].astype(np.complex128)
    scale = np.nanmax(np.abs(meas_c))                               # the largest measured |S_v(f_c)| on the path
    rift_c, meas_c = rift_c / scale, meas_c / scale
    px = axes_in(3.92, 1.30, 1.44, 0.44)
    px.plot(az, rift_c.real, color=ACCENT, linewidth=0.8, zorder=2, label="RIFT, real")
    px.plot(az, rift_c.imag, color=ACCENT, linewidth=0.9, linestyle=(0, (1, 1.1)), zorder=2, label="RIFT, imaginary")
    train, held = ring_roles == "train", ring_roles == "validation"
    for part in (np.real, np.imag):
        px.scatter(az[train], part(meas_c[train]), s=3.5, color=BASE, linewidths=0, zorder=3,
                   label="training" if part is np.real else None)
        px.scatter(az[held], part(meas_c[held]), s=7, facecolors="white", edgecolors=INK, linewidths=0.5, zorder=4,
                   label="held out" if part is np.real else None)
    az_star = az[views == data["v_star"]][0]
    px.axvline(az_star, color=MUTED, linewidth=0.4, linestyle=(0, (1, 1.5)), zorder=1)
    px.text(az_star + 1.5, 0.03, "$v^\\star$", transform=px.get_xaxis_transform(), ha="left", va="bottom",
            fontsize=6.8, color=INK)
    px.set_xlim(*WINDOW)
    px.set_xticks([-60, -40, -20, 0, 20])
    lim = 1.08 * max(np.abs(rift_c.real).max(), np.abs(rift_c.imag).max(), 1.0)
    px.set_ylim(-lim, lim)
    px.set_yticks([-1, 0, 1])
    px.set_xlabel("azimuth along the path (deg; nose at 0)", fontsize=6.2, labelpad=1)
    px.set_ylabel("$S_v(f_c)$", fontsize=6.4, labelpad=1)
    px.tick_params(length=2, pad=1, labelsize=5.8)
    for side in ("top", "right"):
        px.spines[side].set_visible(False)
    handles, labels = px.get_legend_handles_labels()
    px.legend(handles, labels, frameon=False, fontsize=5.5, loc="lower left", bbox_to_anchor=(-0.06, 1.02), ncol=2,
              handlelength=1.4, handletextpad=0.3, columnspacing=0.9, borderaxespad=0, labelspacing=0.15)

    label(3.66, 0.95, "inverse problem: 3D reconstruction", va="center")
    arrow(fig, (bx0 + bw + 0.03, 0.62), (3.61, 0.93), color=ACCENT, rad=0.2)
    mesh_tile = mpimg.imread(PANELS / "b787_mesh_view3.png")[..., :3]
    mip_tile = mpimg.imread(PANELS / "b787_rift_mip_view3.png")[..., :3]
    a = crop_box(mesh_tile, level=0.2, pad=0.06)
    b = crop_box(mip_tile, level=0.08, pad=0.06)
    box = (min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3]))   # one crop: same camera, same scale
    side = 0.80
    for i, (image, name) in enumerate(((mesh_tile, "reference mesh"), (mip_tile, "learned field"))):
        t = axes_in(3.72 + i * (side + 0.10), 0.04, side, side * (box[1] - box[0]) / (box[3] - box[2]))
        tile(t, image, box)
        t.text(0.04, 0.96, name, transform=t.transAxes, ha="left", va="top", fontsize=5.8, color="white")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / "rift_schematic_draft"
    for ext in args.formats.split(","):
        fig.savefig(f"{stem}.{ext}", dpi=400)
        if ext == "svg":
            retarget_svg_family(f"{stem}.svg")
    print("wrote", stem, "| v* =", int(data["v_star"]), "| ring views", len(az))


if __name__ == "__main__":
    main()
