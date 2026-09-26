#!/usr/bin/env python
"""Visualize B787 radar NVS over the Fibonacci view sphere.

The range-power evaluator stores one predicted signal statistic and one error
statistic per held-out direction.  This script additionally derives the same
measured statistic for all 2,000 views directly from the raw responses.  It
then performs sphere-aware inverse-distance interpolation (nearest neighbours
are found in 3D unit-vector space, so the longitude seam and poles are handled
correctly) and draws:

* the 1,800 training and 200 held-out directions together in 3D and on a 2D
  Mollweide sphere, plus the dense measured signal reference; and
* for every supplied method, a dense NVS composite made from measured signal
  at training directions and predictions at held-out directions, alongside
  validation-only normalized range-power relative MSE.

No interpolation enters any reported metric; it is visualization only.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import numpy as np
from scipy.spatial import cKDTree


DEFAULT_CACHES = (
    "RIFT · Round 6 best=figures/b787_range_power/rift_r6_occ_dc_per_view.npz",
    "RIFT · Round 5 best=figures/b787_range_power/rift_r5_target20k_per_view.npz",
    "SpINR-style baseline=figures/b787_range_power/spinr_style_deg0_per_view.npz",
)

LIGHT_SPEED = 299_792_458.0


def parse_cache(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"cache must be LABEL=PATH, got {spec!r}")
    label, path = spec.split("=", 1)
    return label, Path(path)


def load_cache(label: str, path: Path) -> dict[str, np.ndarray | str]:
    with np.load(path) as saved:
        data = {key: saved[key] for key in saved.files}
    complete = np.isfinite(data["range_power_rel_mse"])
    if not complete.all():
        raise RuntimeError(f"{path} has only {complete.sum()}/{len(complete)} completed views")
    data["label"] = label
    return data


def fixed_split(
    num_views: int,
    num_train: int = 1800,
    num_val: int = 200,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the evaluator's fixed seed-42 train/tail-validation split."""

    if num_train + num_val > num_views:
        raise ValueError(f"requested {num_train + num_val} views from {num_views}")
    permutation = np.random.default_rng(seed).permutation(num_views)
    train = permutation[:num_train]
    val = permutation[num_views - num_val :] if num_val else permutation[:0]
    return train, val


def load_or_compute_target_cache(
    raw: np.lib.npyio.NpzFile,
    cache_path: Path,
    stats_path: Path,
    validation_data: dict[str, np.ndarray | str],
    extent: float = 0.15,
    margin: float = 0.05,
) -> dict[str, np.ndarray]:
    """Load or derive measured normalized range-power signal for every view."""

    positions = np.asarray(raw["viewpoint_positions"], dtype=np.float64)
    train_indices, val_indices = fixed_split(len(positions))
    expected_val = np.asarray(validation_data["view_indices"], dtype=np.int64)
    if not np.array_equal(val_indices, expected_val):
        raise RuntimeError("held-out cache does not match the fixed seed-42 tail split")

    if cache_path.exists():
        with np.load(cache_path) as saved:
            cached = {key: saved[key] for key in saved.files}
        cache_valid = (
            np.array_equal(cached.get("view_indices"), np.arange(len(positions)))
            and np.array_equal(cached.get("train_indices"), train_indices)
            and np.array_equal(cached.get("val_indices"), val_indices)
            and np.asarray(cached.get("target_signal", [])).shape == (len(positions),)
            and np.isfinite(cached.get("target_signal", np.array([]))).all()
        )
        if cache_valid:
            target_signal = np.asarray(cached["target_signal"], dtype=np.float64)
        else:
            raise RuntimeError(f"existing target cache has incompatible contents: {cache_path}")
    else:
        with stats_path.open("r", encoding="utf-8") as handle:
            stats = json.load(handle)
        peak_power = float(stats["peak_power"])
        dynamic_range_db = float(stats["dynamic_range_db"])
        metadata = json.loads(str(raw["metadata_json"]))
        num_freq = int(metadata["num_adc_samples"])
        range_step = LIGHT_SPEED / (2.0 * float(metadata["radar_bandwidth_hz"]))
        ranges = np.arange(num_freq, dtype=np.float64) * range_step
        floor = 10.0 ** (-dynamic_range_db / 10.0)
        response = raw["response"]
        target_signal = np.empty(len(positions), dtype=np.float64)

        print(f"deriving measured range-power signal for {len(positions)} views", flush=True)
        for view_index in range(len(positions)):
            measured = response[view_index].mean(axis=2).reshape(-1, num_freq)
            target_power = np.abs(np.fft.ifft(measured, axis=-1)) ** 2
            relative = target_power / peak_power
            db = 10.0 * np.log10(np.maximum(relative, floor))
            target_intensity = np.clip(
                (db + dynamic_range_db) / dynamic_range_db, 0.0, 1.0
            )
            center_range = float(np.linalg.norm(positions[view_index]))
            radius = np.sqrt(3.0) * extent + margin
            roi = (ranges >= center_range - radius) & (ranges <= center_range + radius)
            if not roi.any():
                raise RuntimeError(f"empty range ROI for view {view_index}")
            target_signal[view_index] = float(target_intensity[:, roi].mean())
            if (view_index + 1) % 250 == 0 or view_index + 1 == len(positions):
                print(f"  measured views {view_index + 1}/{len(positions)}", flush=True)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name(cache_path.name + f".tmp.{os.getpid()}.npz")
        np.savez_compressed(
            temporary,
            view_indices=np.arange(len(positions), dtype=np.int64),
            viewpoint_positions=positions,
            target_signal=target_signal,
            train_indices=train_indices,
            val_indices=val_indices,
            peak_power=np.asarray(peak_power),
            dynamic_range_db=np.asarray(dynamic_range_db),
            extent=np.asarray(extent),
            margin=np.asarray(margin),
        )
        os.replace(temporary, cache_path)
        print(f"wrote {cache_path}", flush=True)

    validation_target = np.asarray(validation_data["target_signal"], dtype=np.float64)
    max_difference = float(np.max(np.abs(target_signal[val_indices] - validation_target)))
    if not np.allclose(target_signal[val_indices], validation_target, rtol=1.0e-5, atol=1.0e-7):
        raise RuntimeError(
            "all-view measured signal does not reproduce the evaluator cache; "
            f"max absolute difference={max_difference:.3e}"
        )
    print(f"validated all-view target cache against held-out evaluator (max |Δ|={max_difference:.3e})")
    return {
        "view_indices": np.arange(len(positions), dtype=np.int64),
        "viewpoint_positions": positions,
        "target_signal": target_signal,
        "train_indices": train_indices,
        "val_indices": val_indices,
    }


def unit_vectors(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return points / np.linalg.norm(points, axis=-1, keepdims=True)


def lon_lat(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unit = unit_vectors(points)
    return np.arctan2(unit[:, 1], unit[:, 0]), np.arcsin(np.clip(unit[:, 2], -1.0, 1.0))


def map_grid(n_lon: int = 361, n_lat: int = 181):
    lon = np.linspace(-np.pi, np.pi, n_lon)
    lat = np.linspace(-np.pi / 2.0, np.pi / 2.0, n_lat)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    query = np.stack(
        (
            np.cos(lat_grid) * np.cos(lon_grid),
            np.cos(lat_grid) * np.sin(lon_grid),
            np.sin(lat_grid),
        ),
        axis=-1,
    ).reshape(-1, 3)
    return lon_grid, lat_grid, query


def spherical_idw(
    positions: np.ndarray,
    values: np.ndarray,
    query: np.ndarray,
    shape: tuple[int, int],
    neighbours: int = 8,
    log_values: bool = False,
) -> np.ndarray:
    source = unit_vectors(positions)
    values = np.asarray(values, dtype=np.float64)
    if log_values:
        values = np.log10(np.clip(values, 1.0e-12, None))
    distance, index = cKDTree(source).query(query, k=min(neighbours, len(source)), workers=-1)
    if distance.ndim == 1:
        distance, index = distance[:, None], index[:, None]
    weight = 1.0 / np.maximum(distance, 1.0e-6) ** 2
    interpolated = (weight * values[index]).sum(axis=1) / weight.sum(axis=1)
    if log_values:
        interpolated = 10.0 ** interpolated
    return interpolated.reshape(shape)


def style_mollweide(ax, title: str) -> None:
    ax.set_title(title, fontsize=11.5, pad=12)
    ax.grid(True, linewidth=0.45, alpha=0.32, color="#5c5a56")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_facecolor("#f5f4f1")


def global_rel_mse(data: dict[str, np.ndarray | str]) -> float:
    return float(np.asarray(data["sq_error_db"]).sum() / np.asarray(data["target_sq_db"]).sum())


def plot_sampling(
    target_data: dict[str, np.ndarray],
    out_path: Path,
    neighbours: int,
) -> None:
    all_positions = np.asarray(target_data["viewpoint_positions"])
    all_target = np.asarray(target_data["target_signal"])
    train_indices = np.asarray(target_data["train_indices"], dtype=np.int64)
    held_indices = np.asarray(target_data["val_indices"], dtype=np.int64)
    train_positions = all_positions[train_indices]
    held_positions = all_positions[held_indices]
    train_lon, train_lat = lon_lat(train_positions)
    held_lon, held_lat = lon_lat(held_positions)
    lon_grid, lat_grid, query = map_grid()
    target_map = spherical_idw(
        all_positions, all_target, query, lon_grid.shape, neighbours=neighbours
    )
    signal_lo, signal_hi = np.percentile(all_target, (1.0, 99.0))
    signal_norm = Normalize(signal_lo, signal_hi)

    fig = plt.figure(figsize=(15.4, 4.8), facecolor="#fcfcfb")
    ax3d = fig.add_subplot(1, 3, 1, projection="3d")
    train_unit = unit_vectors(train_positions)
    held_unit = unit_vectors(held_positions)
    ax3d.scatter(train_unit[:, 0], train_unit[:, 1], train_unit[:, 2], s=2.8,
                 color="#4c78a8", alpha=0.30, linewidths=0, label="train · 1,800")
    ax3d.scatter(held_unit[:, 0], held_unit[:, 1], held_unit[:, 2], s=11,
                 color="#ef8354", alpha=0.90, linewidths=0, label="held out · 200")
    ax3d.set_box_aspect((1, 1, 1))
    ax3d.set_axis_off()
    ax3d.view_init(elev=20, azim=33)
    ax3d.set_title("3D Fibonacci acquisition sphere\n1,800 train + 200 held out",
                   fontsize=11.5, pad=10)
    ax3d.legend(loc="lower center", bbox_to_anchor=(0.5, -0.04), frameon=False,
                fontsize=8.5, ncol=2, markerscale=1.4)

    ax_proj = fig.add_subplot(1, 3, 2, projection="mollweide")
    ax_proj.scatter(train_lon, train_lat, s=3.0, c="#4c78a8", alpha=0.38,
                    linewidths=0, rasterized=True, label="train · 1,800")
    ax_proj.scatter(held_lon, held_lat, s=12, c="#ef8354", alpha=0.92,
                    linewidths=0, rasterized=True, label="held out · 200")
    style_mollweide(ax_proj, "Train + held-out directions\nMollweide projection")
    ax_proj.legend(loc="lower center", bbox_to_anchor=(0.5, -0.10), frameon=False,
                   fontsize=8.2, ncol=2, markerscale=1.3)

    ax_target = fig.add_subplot(1, 3, 3, projection="mollweide")
    image = ax_target.pcolormesh(lon_grid, lat_grid, target_map, shading="auto",
                                 cmap="viridis", norm=signal_norm, rasterized=True)
    ax_target.scatter(train_lon, train_lat, s=1.5, c="white", alpha=0.13, linewidths=0)
    ax_target.scatter(held_lon, held_lat, s=8, facecolors="none", edgecolors="#ef8354",
                      alpha=0.78, linewidths=0.45)
    style_mollweide(ax_target, "Measured train + held-out signal\n2,000-view dense reference")
    colorbar = fig.colorbar(image, ax=ax_target, orientation="horizontal",
                            fraction=0.055, pad=0.09)
    colorbar.set_label("mean normalized 60 dB range intensity", fontsize=9)

    fig.suptitle("Radar novel-view synthesis is a function on the viewing sphere",
                 fontsize=15, y=0.995)
    fig.text(0.5, 0.012,
             "Blue = 1,800 measured training views; orange = 200 held-out views from the fixed seed-42 permutation tail. "
             "The dense signal reference uses measurements at all 2,000 views; interpolation is visualization only.",
             ha="center", va="bottom", fontsize=8.5, color="#66635e")
    fig.tight_layout(rect=[0, 0.045, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_comparison(
    datasets: list[dict[str, np.ndarray | str]],
    target_data: dict[str, np.ndarray],
    out_path: Path,
    neighbours: int,
) -> None:
    lon_grid, lat_grid, query = map_grid()
    all_positions = np.asarray(target_data["viewpoint_positions"])
    train_indices = np.asarray(target_data["train_indices"], dtype=np.int64)
    train_positions = all_positions[train_indices]
    train_signal = np.asarray(target_data["target_signal"])[train_indices]
    train_lon, train_lat = lon_lat(train_positions)
    all_signals = [train_signal]
    all_signals.extend(np.asarray(data["pred_signal"]) for data in datasets)
    signal_values = np.concatenate(all_signals)
    signal_lo, signal_hi = np.percentile(signal_values, (1.0, 99.0))
    signal_norm = Normalize(signal_lo, signal_hi)

    all_errors_pct = np.concatenate(
        [100.0 * np.asarray(data["range_power_rel_mse"]) for data in datasets]
    )
    positive = all_errors_pct[all_errors_pct > 0]
    error_lo = max(float(np.percentile(positive, 1.0)), 1.0e-3)
    error_hi = max(float(np.percentile(positive, 99.0)), error_lo * 10.0)
    error_norm = LogNorm(error_lo, error_hi)

    n_methods = len(datasets)
    fig, axes = plt.subplots(
        2,
        n_methods,
        figsize=(4.45 * n_methods, 7.0),
        subplot_kw={"projection": "mollweide"},
        squeeze=False,
    )
    fig.patch.set_facecolor("#fcfcfb")
    signal_image = error_image = None

    for column, data in enumerate(datasets):
        held_positions = np.asarray(data["viewpoint_positions"])
        held_lon, held_lat = lon_lat(held_positions)
        pred_signal = np.asarray(data["pred_signal"])
        error_pct = 100.0 * np.asarray(data["range_power_rel_mse"])
        composite_positions = np.concatenate((train_positions, held_positions), axis=0)
        composite_signal = np.concatenate((train_signal, pred_signal), axis=0)
        pred_map = spherical_idw(
            composite_positions,
            composite_signal,
            query,
            lon_grid.shape,
            neighbours=neighbours,
        )
        error_map = spherical_idw(
            held_positions,
            error_pct,
            query,
            lon_grid.shape,
            neighbours=neighbours,
            log_values=True,
        )

        ax_signal, ax_error = axes[0, column], axes[1, column]
        signal_image = ax_signal.pcolormesh(
            lon_grid, lat_grid, pred_map, shading="auto", cmap="viridis",
            norm=signal_norm, rasterized=True,
        )
        error_image = ax_error.pcolormesh(
            lon_grid, lat_grid, error_map, shading="auto", cmap="magma",
            norm=error_norm, rasterized=True,
        )
        ax_signal.scatter(train_lon, train_lat, s=1.5, c="white", alpha=0.11,
                          linewidths=0)
        ax_signal.scatter(held_lon, held_lat, s=7.0, facecolors="none",
                          edgecolors="#ef8354", alpha=0.72, linewidths=0.4)
        ax_error.scatter(held_lon, held_lat, s=2.0, c="white", alpha=0.27, linewidths=0)
        label = str(data["label"])
        style_mollweide(
            ax_signal,
            f"{label}\nmeasured train + predicted held out",
        )
        style_mollweide(
            ax_error,
            f"{label}\nper-view error · global {100.0 * global_rel_mse(data):.3f}%",
        )

    fig.subplots_adjust(left=0.025, right=0.88, bottom=0.065, top=0.91,
                        hspace=0.23, wspace=0.10)
    signal_cax = fig.add_axes((0.895, 0.555, 0.012, 0.315))
    error_cax = fig.add_axes((0.895, 0.13, 0.012, 0.315))
    cbar_signal = fig.colorbar(signal_image, cax=signal_cax, orientation="vertical")
    cbar_signal.set_label("mean normalized 60 dB range intensity", fontsize=9.5)
    cbar_error = fig.colorbar(error_image, cax=error_cax, orientation="vertical")
    cbar_error.set_label("normalized range-power relative MSE [%] · log scale", fontsize=9.5)
    fig.suptitle("B787 NVS composite: 1,800 measured train + 200 predicted held-out views",
                 fontsize=15, y=0.986)
    fig.text(0.5, 0.011,
             "Top: measured training values plus model predictions only at held-out views (orange rings). "
             "Bottom: error only at the 200 held-out views. Spherical IDW is visualization only; metrics use observed validation points.",
             ha="center", va="bottom", fontsize=8.5, color="#66635e")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz-path", default="data/b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere2k.npz")
    parser.add_argument(
        "--stats",
        default="training_checkpoints/b787_radar_fields_released/radar_fields_power_stats.json",
    )
    parser.add_argument(
        "--target-cache",
        default="figures/b787_range_power/b787_target_signal_all_views.npz",
    )
    parser.add_argument("--caches", nargs="+", default=list(DEFAULT_CACHES),
                        help="one or more LABEL=per_view.npz specifications")
    parser.add_argument("--sampling-out", default="figures/b787_range_power/fibonacci_sampling.png")
    parser.add_argument("--comparison-out", default="figures/b787_range_power/nvs_sphere_comparison.png")
    parser.add_argument("--neighbours", type=int, default=8)
    args = parser.parse_args()

    parsed = [parse_cache(spec) for spec in args.caches]
    datasets = [load_cache(label, path) for label, path in parsed]
    with np.load(args.npz_path, allow_pickle=True) as raw:
        target_data = load_or_compute_target_cache(
            raw,
            Path(args.target_cache),
            Path(args.stats),
            datasets[0],
        )
    plot_sampling(target_data, Path(args.sampling_out), args.neighbours)
    plot_comparison(datasets, target_data, Path(args.comparison_out), args.neighbours)
    print("wrote", args.sampling_out)
    print("wrote", args.comparison_out)


if __name__ == "__main__":
    main()
