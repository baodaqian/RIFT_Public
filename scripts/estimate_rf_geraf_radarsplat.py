#!/usr/bin/env python3
"""Reproduce explicitly assumed scenarios in parallization_guide.md.

Arithmetic only: no data, Torch, CUDA, model execution or timing measurement.
Service times and sustained rates are INPUT ASSUMPTIONS, not observations,
confidence intervals, demonstrated GPU rankings or bounds on actual runtime.
RF V100 is blocked by the current backend. RadarSplat RIFT epochs exceed its
fixed 2000-update recipe and are extrapolations. Cold work/checkpoint I/O is
excluded; see the guide for validation cadence and preparation formulas.
GeRaF timings are stale: selected 101-cubed lazy targets add per-update MF
corner work absent from the old 601-cubed dense-cache scenarios. Those old
numbers require explicit --legacy-geraf-timing and are not current estimates.

--storage reports logical disk bytes for one collection object or Camry HH.
It excludes source archives, filesystem allocation overhead, logs and exports.
Checkpoint budgets include populated Adam moments, but not transient gradients.
No real trained checkpoint or converted target is created/read by this script.
"""
import argparse
import json


CARDS = ("V100 16 GB", "A100 40 GB", "L40S 48 GB", "H100 80 GB")
TRACE_POINTS = 807 * 64
NATIVE_TRAIN_SAMPLES = 101_543_576
NATIVE_VAL_SAMPLES = 22_338_864
# Billion composite entries/second: (slow scenario, fast scenario).
RANGE_RATES = ((.1, .8), (.2, 1.5), (.08, .6), (.4, 3))
NATIVE_RATES = ((.2, 1.5), (.5, 3), (.05, .4), (1, 6))
# Seconds/update or seconds/validation view, unless explicitly noted.
RF_TRAIN = {
    "RIFT": (None, (.15, 1), (.15, 1), (.10, .8)),
    "GOTCHA HH": (None, (1.1, 9), (1.1, 10), (1, 8.5)),
}
RF_VAL = {
    "RIFT": (None, (.025, .15), (.025, .18), (.020, .12)),
    # Seconds/pulse; add 0.001–0.015 seconds/sector below.
    "GOTCHA HH": (None, (.001, .008), (.0012, .010), (.0008, .007)),
}
RS_TRAIN = {
    "RIFT": ((.20, 4), (.10, 2), (.08, 2), (.06, 1.5)),
    "GOTCHA HH": ((.10, 2), (.05, 1), (.04, 1), (.03, .8)),
}
RS_VAL = {
    "RIFT": ((.10, 2), (.05, 1), (.04, 1), (.03, .75)),
    "GOTCHA HH": ((.05, 1), (.025, .5), (.02, .5), (.015, .4)),
}


def estimate_rows(*, legacy_geraf=False):
    rows = []

    def add(method, dataset, card, train, val, status):
        rows.append(dict(method=method, dataset=dataset, gpu=card,
                         train_seconds=train, validation_pass_seconds=val,
                         status=status))

    for dataset, steps, nval in (("RIFT", 320, 1000), ("GOTCHA HH", 200, 440)):
        for i, card in enumerate(CARDS):
            if i == 0:
                add("Radar Fields", dataset, card, None, None,
                    "blocked: current SM70/TCNN backend")
                continue
            train = [steps * x for x in RF_TRAIN[dataset][i]]
            val = ([nval * x for x in RF_VAL[dataset][i]] if dataset == "RIFT"
                   else [52_100 * x + nval * y for x, y in
                         zip(RF_VAL[dataset][i], (.001, .015))])
            add("Radar Fields", dataset, card, train, val,
                "assumed service-time scenario")

    for dataset, ntrain, nval in (("RIFT", 3200, 1000), ("GOTCHA HH", 2000, 440)):
        wt = (3 * TRACE_POINTS * 256 * 20 * ntrain if dataset == "RIFT"
              else 3 * TRACE_POINTS * NATIVE_TRAIN_SAMPLES)
        wv = (2 * TRACE_POINTS * 256 * 20 * nval if dataset == "RIFT"
              else 2 * TRACE_POINTS * NATIVE_VAL_SAMPLES)
        for i, card in enumerate(CARDS):
            slow, fast = (RANGE_RATES if dataset == "RIFT" else NATIVE_RATES)[i]
            train = [wt / (fast * 1e9) + ntrain * .5,
                     wt / (slow * 1e9) + ntrain * 6]
            val = [wv / (fast * 1e9) + nval * .5,
                   wv / (slow * 1e9) + nval * 6]
            add("GeRaF", dataset, card, train, val,
                "unmasked workload plus assumed rate; warm banks")

    for dataset, ntrain, nval in (("RIFT", 3200, 1000), ("GOTCHA HH", 2000, 440)):
        for i, card in enumerate(CARDS):
            add("RadarSplat", dataset, card,
                [ntrain * x for x in RS_TRAIN[dataset][i]],
                [nval * x for x in RS_VAL[dataset][i]],
                "conditional raster service; " +
                ("RIFT epoch extrapolated beyond recipe" if dataset == "RIFT"
                 else "2000-update recipe"))
    if not legacy_geraf:
        for item in rows:
            if item['method'] == 'GeRaF':
                item.update(train_seconds=None, validation_pass_seconds=None,
                            status='stale: 101^3 lazy-target MF work is uncalibrated')
    return rows


def format_interval(values, unit):
    if values is None:
        return "Blocked"
    divisor = 3600 if unit == "h" else 60
    return f"{values[0] / divisor:.2f}–{values[1] / divisor:.2f} {unit}"


def markdown_table(rows, method):
    lines = ["| Dataset | GPU | Training epoch scenario | One validation pass |",
             "| --- | --- | ---: | ---: |"]
    unit = "h" if method == "GeRaF" else "min"
    for row in rows:
        if row["method"] == method:
            stale = row["status"].startswith("stale:")
            lines.append("| " + " | ".join((row["dataset"], row["gpu"],
                         "Stale: lazy MF" if stale else format_interval(row["train_seconds"], unit),
                         "Stale: lazy MF" if stale else format_interval(row["validation_pass_seconds"], unit))) + " |")
    return "\n".join(lines)


def geraf_lazy_workload(mf_grid=101):
    """Work bounds only, not wall times: dedup/zero-padding can reduce corners.

    InterpolateMFAtTargets consumes 64 target samples per <=807 rays before
    loss masking. Compute MF at <=8 valid corners/query, deduplicate, then
    interpolate the FP32 magnitudes, preserving dense-grid semantics at the
    selected resolution. Corner MF is additional to model prediction physics.
    The single accumulated volume still requires one full lattice MF per
    training view at preparation; no validation cube is prepared/persisted.
    """
    if type(mf_grid) is not int or mf_grid < 2:
        raise ValueError("GeRaF MF grid must be an integer >= 2")
    corners = min(8 * TRACE_POINTS, mf_grid**3)
    rows = []
    for dataset, nt, nv, train_entries, val_entries in (
            ("RIFT", 3200, 1000, 3200 * 256 * 20, 1000 * 256 * 20),
            ("GOTCHA HH", 2000, 440, NATIVE_TRAIN_SAMPLES, NATIVE_VAL_SAMPLES)):
        rows.append(dict(dataset=dataset, mf_grid=mf_grid,
                         target_queries_per_view_upper=TRACE_POINTS,
                         unique_mf_corners_per_view_upper=corners,
                         unit="range tap" if dataset == "RIFT" else "native phase",
                         extra_target_work_per_train_epoch_upper=corners * train_entries,
                         extra_target_work_per_validation_pass_upper=corners * val_entries,
                         accumulated_preparation_work=mf_grid**3 * train_entries,
                         partial_accumulation_save_count=nt,
                         partial_accumulation_tensor_write_bytes=nt * 4 * mf_grid**3,
                         timing_status="uncalibrated; measure corner MF/dedup/interpolation, preparation and recovery I/O"))
    return rows


def storage_rows(geraf_mf_grid=101, geraf_cache_mode="lazy"):
    """Source-shape arithmetic plus explicitly rounded serialization budgets.

    Audited 2026-09-20 against:
      train_radar_fields.py::checkpoint_payload / atomic_torch_save;
      rift/radar_fields_gotcha.py::run_gotcha;
      rift/geraf_source_training.py::SourceTargets / train;
      rift/vendor/geraf_sens/rf_rendering.py::state_dict;
      rift/radarsplat_release_training.py::train;
      rift/radarsplat_gotcha.py::GOTCHAPowerCache.

    RadarSplat NPZ sizes include uncompressed NumPy/ZIP headers and role text.
    RIFT uses float32 calibration; native GOTCHA uses float64 calibration.
    A metadata-only CPU probe using A320/the registered native split measured
    3,378,114 bytes for RIFT acquisition.npz. Synthetic populated-Adam model
    serialization INCLUDING acquisition metadata measured 33,727,045 bytes
    (RIFT) / 34,657,093 (GOTCHA), before remaining recipe/sampler/history fields.
    Those are serialization probes, not production checkpoints or CUDA tests.
    The rounded 35 MB budget allows for the remaining fields.

    RF cannot be instantiated without allocated TCNN: use the conservative
    16*2*2**19 + 32768 parameter ceiling and 12 bytes/parameter for FP32
    weights plus two Adam moments. Include <= 8*48**3 readout bytes on RIFT,
    then round to 210 MB/file for buffers, protocol/coverage metadata and ZIP.
    This is a planning allowance, not a measured checkpoint file size.

    GeRaF banks are float64 real+imag (16 bytes/native complex sample), not
    two complete copies for its two alternating banks. All visited train views
    are retained, no validation banks. At full exposure add 596616*12 bytes
    model/moments, then budget 7.9 GB (RIFT) / 1.64 GB (GOTCHA) per file for
    tensor/container/protocol overhead. Earlier best checkpoints may be smaller.
    Selected GeRaF policy: target_storage=lazy_trilinear_accumulated_only_v1,
    mf_grid=101. Persist one accumulated volume, no per-view cubes. Atomic
    per-view preparation recovery retains two grid payloads at peak (old/new
    progress, or progress/final), plus small metadata. Historical 'dense' mode
    here is a calculator option ONLY: it models train+validation+accumulated
    volumes, not a flag accepted by the new immutable training recipe.
    Grid size changes target/cache work, not bank/checkpoint size. The 101
    supervision lattice is an explicit recipe change, not accuracy equivalence.

    Normal atomic writes require one additional checkpoint-sized temporary.
    Periodic saves overwrite fixed filenames; they do not retain 500 snapshots.
    Abruptly killed writers can leave additional orphan temporaries; excluded.
    Historical dense preparation could have accumulated+view temporaries,
    covered by its completed allocation. Existing incompatible caches are not
    deleted or reused. Source archives and retained old runs are additional.
    """
    if type(geraf_mf_grid) is not int or geraf_mf_grid < 2:
        raise ValueError("GeRaF MF grid must be an integer >= 2")
    if geraf_cache_mode not in ("lazy", "dense"):
        raise ValueError("GeRaF calculator cache mode must be lazy or dense")
    result = []

    def row(method, dataset, cache, file_budget, file_count, **details):
        result.append(dict(method=method, dataset=dataset,
                           cache_budget_bytes=cache,
                           checkpoint_file_budget_bytes=file_budget,
                           retained_checkpoint_files=file_count,
                           retained_checkpoint_budget_bytes=file_count * file_budget,
                           atomic_checkpoint_peak_bytes=(file_count + 1) * file_budget,
                           total_cache_plus_atomic_checkpoint_peak_bytes=
                           cache + (file_count + 1) * file_budget, **details))

    rf_core = 12 * (16 * 2 * 2**19 + 32768)
    row("Radar Fields", "RIFT", 200_000, 210_000_000, 3,
        checkpoint_tensor_ceiling_bytes=rf_core + 8 * 48**3,
        cache_note="TRAIN stats JSON (~140 kB); no persistent power images",
        files="best, latest, final; root latest can precede final")
    row("Radar Fields", "GOTCHA HH", 0, 210_000_000, 3,
        checkpoint_tensor_ceiling_bytes=rf_core,
        cache_note="No separate target cache; normalization stats in checkpoint",
        files="best, latest, final")

    volume = 4 * geraf_mf_grid**3 + 128  # FP32 volume plus small .npy v1 header.
    for dataset, nt, nv, ns, budget in (
            ("RIFT", 3200, 1000, 3200 * 256 * 600, 7_900_000_000),
            ("GOTCHA HH", 2000, 440, NATIVE_TRAIN_SAMPLES, 1_640_000_000)):
        lazy = geraf_cache_mode == "lazy"
        cache = volume if lazy else (nt + nv + 1) * volume
        row("GeRaF", dataset, cache, budget, 2,
            checkpoint_core_bytes=16 * ns + 12 * 596616,
            cache_train_bytes=0 if lazy else nt * volume,
            cache_validation_bytes=0 if lazy else nv * volume,
            cache_accumulated_bytes=volume,
            cache_preparation_peak_payload_bytes=2 * volume if lazy else cache,
            cpu_response_bank_bytes=16 * ns,
            volume_bytes=volume,
            mf_grid=geraf_mf_grid,
            cache_mode=geraf_cache_mode,
            cache_note=("One accumulated volume; no per-view files; partial preparation recovery; metadata extra"
                        if lazy else "Historical dense train+validation+accumulated volumes; metadata extra"),
            files="latest, best (no separate final)")

    for dataset, nt, nv, btrain, bval, cache, acq, core, ram in (
            ("RIFT", 3200, 1000, 7080, 7100, 34_000_000, 3_378_114,
             33_727_045, 2000 * 33**2 + 3000 * 33**2 * 8),
            ("GOTCHA HH", 2000, 440, 23302, 23322, 58_000_000, 0,
             34_657_093, 2000 * 33 * 145 * 9)):
        row("RadarSplat", dataset, cache, 35_000_000, 2,
            target_npz_bytes=nt * btrain + nv * bval,
            separate_acquisition_npz_bytes=acq,
            checkpoint_parameter_moment_bytes=20000 * 120 * 12,
            checkpoint_model_moments_calibration_probe_bytes=core,
            derived_cpu_label_background_bytes_upper=ram,
            cache_note="Targets + calibration/recipe/manifests; RAM labels/backgrounds are not saved",
            files="latest, final; native run-control checkpoint is small and additional")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print cases as JSON")
    parser.add_argument("--storage", action="store_true", help="Print six cache/checkpoint storage cases")
    parser.add_argument("--geraf-mf-grid", type=int, default=101,
                        help="Storage calculation only: GeRaF MF lattice side (default 101)")
    parser.add_argument("--geraf-cache-mode", choices=("lazy", "dense"), default="lazy",
                        help="Storage model only; dense is historical, not a trainer setting")
    parser.add_argument("--legacy-geraf-timing", action="store_true",
                        help="Show superseded dense-601 GeRaF timing scenarios, not lazy-target estimates")
    args = parser.parse_args()
    if args.geraf_mf_grid < 2:
        parser.error("--geraf-mf-grid must be >= 2")
    if (args.geraf_mf_grid != 101 or args.geraf_cache_mode != "lazy") and not args.storage:
        parser.error("GeRaF grid/cache options apply only with --storage; they do not recalibrate timing")
    if args.storage and args.legacy_geraf_timing:
        parser.error("--legacy-geraf-timing does not apply to --storage")
    if args.storage:
        rows = storage_rows(args.geraf_mf_grid, args.geraf_cache_mode)
        lazy_work = geraf_lazy_workload(args.geraf_mf_grid) if args.geraf_cache_mode == "lazy" else []
        if args.json:
            print(json.dumps(dict(evidence="Shape arithmetic and rounded serialization budgets; not measured production files",
                                  units="bytes, decimal MB/GB/TB", rows=rows,
                                  geraf_lazy_workload=lazy_work), indent=2))
        else:
            print("Logical disk budgets; one object/HH run, excluding source archives.")
            for row in rows:
                print(f"{row['method']} / {row['dataset']}: "
                      f"cache {row['cache_budget_bytes'] / 1e9:.6f} GB; "
                      f"checkpoints {row['retained_checkpoint_files']} x "
                      f"{row['checkpoint_file_budget_bytes'] / 1e9:.3f} GB; "
                      f"checkpoint atomic peak {row['atomic_checkpoint_peak_bytes'] / 1e9:.3f} GB")
            for item in lazy_work:
                print(f"GeRaF / {item['dataset']}: <= {item['unique_mf_corners_per_view_upper']:,} "
                      f"new MF corners/view; <= {item['extra_target_work_per_train_epoch_upper']:,} "
                      f"additional {item['unit']} entries/train epoch; timing uncalibrated.")
        return
    rows = estimate_rows(legacy_geraf=args.legacy_geraf_timing)
    if args.json:
        print(json.dumps(dict(evidence="Unmeasured assumptions; see parallization_guide.md",
                              geraf_timing_policy=("historical_dense_601" if args.legacy_geraf_timing
                                                   else "stale_for_selected_lazy_101"),
                              rows=rows), indent=2))
    else:
        print("Uncalibrated scenarios; not measured timings or actual runtime bounds.")
        if args.legacy_geraf_timing:
            print("WARNING: GeRaF rows below are historical dense-601 scenarios, not the selected lazy-101 recipe.")
        for method in ("Radar Fields", "GeRaF", "RadarSplat"):
            print(f"\n{method}\n{markdown_table(rows, method)}")


if __name__ == "__main__":
    main()
