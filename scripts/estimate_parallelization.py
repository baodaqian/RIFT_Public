#!/usr/bin/env python3
"""Reproduce the *uncalibrated scenarios* in parallization_guide.md.

Pure arithmetic: no Torch, CUDA, data access, fitting, or timing measurement.
Work counts are from the registered metadata/source audit on 2026-09-20.
Traffic, equivalent operations, launch costs and efficiencies are ASSUMPTIONS.
These are neither confidence intervals nor lower/upper bounds on actual time.
"""
from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True)
class GPU:
    name: str
    gbps: float
    fp64: float
    fp32: float
    rift_tile: int
    neural_tile: int


GPUS = (
    GPU("V100 16 GB (PCIe)", 900, 7, 14, 49152, 368640),
    GPU("A30 24 GB", 933, 5.2, 10.3, 77824, 577536),
    GPU("A100 40 GB", 1555, 9.7, 19.5, 135168, 884736),
    GPU("A40 48 GB", 696, 37.4 / 64, 37.4, 159744, 884736),
    GPU("L40S 48 GB", 864, 91.6 / 64, 91.6, 159744, 884736),
    GPU("H100 80 GB (SXM)", 3350, 34, 67, 262144, 884736),
    GPU("H200 141 GB (SXM)", 4800, 34, 67, 262144, 884736),
)
# (bandwidth utilization, equivalent-FP64 utilization, FP32 GEMM utilization,
#  effective host/launch cost). Deliberately broad, unmeasured scenarios.
SCENARIOS = ((0.60, 0.25, 0.60, 8e-6), (0.20, 0.05, 0.20, 25e-6))
# (traffic bytes / entry, equivalent FP64 operations / entry,
#  launch-equivalents / Python physics block), for ONE forward-equivalent.
# These proxies include intermediates, NOT resident VRAM or measured DRAM bytes.
COST = {
    "range": (4096, 2048, 400),  # 20-offset scatter plus FFT allowance
    "native": (96, 128, 16),    # direct phase/exponential/reduction
    "direct_bins": (512, 512, 60),  # two sinc evaluations + carrier + gather
    "native_fft": (192, 256, 24),  # native nonaffine phase + pointwise FFT
    "se": (96, 128, 16),        # Fourier phase + weighted reduction
}
Q = 7077888
GOTCHA_POINTS = (48 * 2) ** 3  # User-selected G48/GL2; collection stays G96.
NN_WEIGHTS = 3561600  # dense weights; biases omitted from GEMM count
TRAIN_SAMPLES = 101543576
VAL_SAMPLES = 22338864
TRAIN_PULSES = 236826
VAL_PULSES = 52100
# Sum pulse_count * ceil(GOTCHA_POINTS / (1048576 // native_frequency_count))
# over the eight registered HH passes, including each pass's partial last tile.
TRAIN_NATIVE_BLOCKS = 85823639
VAL_NATIVE_BLOCKS = 18880592
TRAIN_SE_BLOCKS = 7312297
VAL_SE_BLOCKS = 1608656


def physics(gpu, scenario, kind, entries, blocks):
    bw_eff, dp_eff, _, launch_s = scenario
    traffic, operations, launches = COST[kind]
    device = max(entries * traffic / (bw_eff * gpu.gbps * 1e9),
                 entries * operations / (dp_eff * gpu.fp64 * 1e12))
    return device + blocks * launches * launch_s


def neural(gpu, scenario, updates, tile, *, training=True, points=Q):
    _, _, sp_eff, launch_s = scenario
    # Field evaluation + replay forward + approximately two forwards of backward.
    multiplier, launches = (4, 64) if training else (1, 16)
    return (updates * multiplier * points * 2 * NN_WEIGHTS / (sp_eff * gpu.fp32 * 1e12)
            + updates * ceil(points / tile) * launches * launch_s)


def training(gpu, scenario, model, dataset, *, capacity=False):
    if model == "RIFT":
        n = 262144 if capacity else (110592 if dataset == "Collection" else 32768)
        if dataset == "Collection":
            # Forward + checkpoint replay + backward proxy, plus 1/16 probe.
            return 3.1875 * physics(gpu, scenario, "range", 3200 * n * 256,
                                    3200 * ceil(n / gpu.rift_tile))
        # Native also differentiates position stats before final backward;
        # an additional angular probe runs on every tenth sector update.
        return 5.3 * physics(gpu, scenario, "native", n * TRAIN_SAMPLES, TRAIN_PULSES)
    if model == "SpINR":
        if dataset == "Collection":
            return (2 * physics(gpu, scenario, "direct_bins", Q * 9012740, 60886336)
                    + neural(gpu, scenario, 800, 4096))
        return (2 * physics(gpu, scenario, "native_fft", GOTCHA_POINTS * TRAIN_SAMPLES, TRAIN_NATIVE_BLOCKS)
                + neural(gpu, scenario, 232, gpu.neural_tile, points=GOTCHA_POINTS))
    # SE: 2 value+gradient calls at 3 forward-equivalents each, 1 line-search
    # value call. This is one outer iteration across all groups, NOT one pass.
    if dataset == "Collection":
        return 7 * physics(gpu, scenario, "se", 343 * 491520000, 3200 * 51)
    return 7 * physics(gpu, scenario, "se", 74088 * TRAIN_SAMPLES, TRAIN_SE_BLOCKS)


def validation(gpu, scenario, model, dataset, *, capacity=False):
    if model == "RIFT":
        n = 262144 if capacity else (110592 if dataset == "Collection" else 32768)
        if dataset == "Collection":
            return physics(gpu, scenario, "range", 1000 * n * 256,
                           1000 * ceil(n / gpu.rift_tile))
        return physics(gpu, scenario, "native", n * VAL_SAMPLES, VAL_PULSES)
    if model == "SpINR":
        if dataset == "Collection":
            # Full 1000-view validation plus 128-view TRAIN diagnostic.
            # Diagnostic selected-entry counts are estimated proportionally.
            direct = physics(gpu, scenario, "direct_bins", Q * 2819771, 19049111)
            full = physics(gpu, scenario, "range", 1000 * Q * 256, 1000 * 1728)
            return 1.128 * (direct + full) + neural(gpu, scenario, 2, 4096, training=False)
        return (physics(gpu, scenario, "native", GOTCHA_POINTS * VAL_SAMPLES, VAL_NATIVE_BLOCKS)
                + neural(gpu, scenario, 1, gpu.neural_tile, training=False, points=GOTCHA_POINTS))
    if dataset == "Collection":
        return physics(gpu, scenario, "se", 343 * 153600000, 1000 * 51)
    return physics(gpu, scenario, "se", 74088 * VAL_SAMPLES, VAL_SE_BLOCKS)


def interval(function, *args, **kwargs):
    a, b = (function(args[0], s, *args[1:], **kwargs) / 3600 for s in SCENARIOS)
    assert 0 < a <= b
    return f"{float(f'{a:.2g}'):g}–{float(f'{b:.2g}'):g} h"


def main():
    print("| Model | Dataset | GPU | Training unit, initial scene | RIFT at capacity |")
    print("| --- | --- | --- | ---: | ---: |")
    rows = 0
    for model in ("RIFT", "SpINR", "SE Stage 1 (conditional)"):
        for dataset in ("Collection", "GOTCHA HH"):
            for gpu in GPUS:
                initial = interval(training, gpu, model, dataset)
                cap = interval(training, gpu, model, dataset, capacity=True) if model == "RIFT" else "—"
                print(f"| {model} | {dataset} | {gpu.name} | {initial} | {cap} |")
                rows += 1
    assert rows == 6 * len(GPUS)
    print("\n| Validation event | " + " | ".join(g.name for g in GPUS) + " |")
    print("| --- | " + " | ".join("---:" for g in GPUS) + " |")
    for model in ("RIFT", "SpINR", "SE Stage 1 (conditional)"):
        for dataset in ("Collection", "GOTCHA HH"):
            values = [interval(validation, g, model, dataset) for g in GPUS]
            print(f"| {model}, {dataset} | " + " | ".join(values) + " |")
    print("\nRIFT validation at full capacity:")
    for dataset in ("Collection", "GOTCHA HH"):
        print(dataset + ": " + "; ".join(interval(validation, g, "RIFT", dataset, capacity=True) for g in GPUS))
    print("\nSE additional line-search sweep (train-unit interval / 7):")
    for dataset in ("Collection", "GOTCHA HH"):
        print(dataset + ": " + "; ".join(
            interval(lambda g, s: training(g, s, "SE", dataset) / 7, g) for g in GPUS))
    print("\nSE Stage 2 GPU/dispatch proxy, seconds per update; excludes CPU refresh:")
    for gpu in GPUS:
        seconds = [6144 * 2 * 1857024 * 10 / (s[2] * gpu.fp32 * 1e12)
                   + 200 * s[3] for s in SCENARIOS]
        print(gpu.name + ": " + "–".join(f"{s:.2g}" for s in seconds)
              + " s; 5000 updates: " + "–".join(f"{s * 5000 / 60:.2g}" for s in seconds) + " min")


if __name__ == "__main__":
    main()
