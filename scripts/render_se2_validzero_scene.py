#!/usr/bin/env python
"""Render the exported SE2 valid-zero surface against the registered B787 outline."""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.render_b787_vs_stl import (  # noqa: E402
    load_stl_vertices,
    stl_into_scene_frame,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", required=True)
    parser.add_argument(
        "--npz-path",
        default="data/b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere2k.npz",
    )
    parser.add_argument("--stl", default="data/B787.stl")
    parser.add_argument("--lim", type=float, default=0.11)
    parser.add_argument("--max-points", type=int, default=8000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with np.load(args.surface, allow_pickle=False) as result:
        points = np.asarray(result["surface_points"], dtype=np.float64)
        policy = str(result["policy"].item())
        manager_identity = str(result["manager_identity"].item())
        if bool(result["ground_truth_geometry_used"].item()):
            raise ValueError("SE2 surface claims geometry truth was used")
        topology = json.loads(str(result["topology_json"].item()))

    rng = np.random.default_rng(0)
    if len(points) > args.max_points:
        points = points[rng.choice(len(points), args.max_points, replace=False)]

    metadata = json.loads(
        str(np.load(args.npz_path, allow_pickle=True, mmap_mode="r")["metadata_json"])
    )
    truth = stl_into_scene_frame(load_stl_vertices(args.stl), metadata)
    if len(truth) > 12000:
        truth = truth[rng.choice(len(truth), 12000, replace=False)]

    fig = plt.figure(figsize=(12.4, 9.2), constrained_layout=True)
    fig.suptitle(
        "Sugavanam–Ertin SE2 valid-zero baseline · v2p1",
        fontsize=16,
        fontweight="semibold",
    )
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax3d.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=points[:, 2], cmap="magma", s=1.6, alpha=0.72, linewidths=0,
    )
    truth_view = truth[rng.choice(len(truth), min(3500, len(truth)), replace=False)]
    ax3d.scatter(
        truth_view[:, 0], truth_view[:, 1], truth_view[:, 2],
        c="#36c9c6", s=0.35, alpha=0.10, linewidths=0,
    )
    ax3d.set_title("Exported zero-level surface")
    ax3d.set_xlabel("x (m)")
    ax3d.set_ylabel("y (m)")
    ax3d.set_zlabel("z (m)")
    ax3d.set_xlim(-args.lim, args.lim)
    ax3d.set_ylim(-args.lim, args.lim)
    ax3d.set_zlim(-args.lim, args.lim)
    ax3d.set_box_aspect((1, 1, 1))
    ax3d.view_init(elev=24, azim=-52)

    projections = [
        ((0, 1), "Top view · x–y"),
        ((0, 2), "Side view · x–z"),
        ((1, 2), "Front view · y–z"),
    ]
    for panel, (axes, title) in enumerate(projections, start=2):
        ax = fig.add_subplot(2, 2, panel)
        horizontal, vertical = axes
        ax.scatter(
            truth[:, horizontal], truth[:, vertical],
            s=0.5, c="#2aa7a4", alpha=0.13, linewidths=0, label="registered STL",
        )
        ax.scatter(
            points[:, horizontal], points[:, vertical],
            s=1.8, c="#d1495b", alpha=0.32, linewidths=0, label="SE2 surface",
        )
        ax.set_title(title)
        ax.set_xlabel(f"{'xyz'[horizontal]} (m)")
        ax.set_ylabel(f"{'xyz'[vertical]} (m)")
        ax.set_xlim(-args.lim, args.lim)
        ax.set_ylim(-args.lim, args.lim)
        ax.set_aspect("equal")
        ax.grid(alpha=0.16, linewidth=0.5)
        if panel == 2:
            ax.legend(loc="upper right", frameon=False, markerscale=4)

    fig.text(
        0.01,
        0.005,
        f"{manager_identity} · policy={policy} · "
        f"closed={topology.get('boundary_edges') == 0} · "
        f"components={topology.get('components')}",
        fontsize=9,
        color="#555555",
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(args.out)


if __name__ == "__main__":
    main()
