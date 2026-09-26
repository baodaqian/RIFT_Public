"""Prepare the LOCAL measured TopHat response screen.

The default command is contract-only and never opens the future archive.  A
caller supplying fake or future shard objects may use ``execute`` to exercise
the metadata-first bind and streamed eight-panel maps locally.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_local(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ACQ = _load_local("gotcha_acquisition", ROOT / "rift" / "gotcha_acquisition.py")
SOURCE_AF = _load_local("gotcha_source_af", ROOT / "rift" / "gotcha_source_af.py")
LOC = _load_local("gotcha_tophat_localization", ROOT / "rift" / "gotcha_tophat_localization.py")


PROTOCOL_PATH = ROOT / "protocols" / "gotcha_step3_tophat_measured_response_screen_v1.json"
POWER_DISPLAY_FLOOR_DB = -80.0


def _power_db(values):
    floor_linear = 10.0 ** (POWER_DISPLAY_FLOOR_DB / 10.0)
    return np.maximum(10.0 * np.log10(np.maximum(np.asarray(values, dtype=np.float64), floor_linear)), POWER_DISPLAY_FLOOR_DB)


def contract_only_result() -> dict[str, Any]:
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    return {
        "status": LOC.MEASURED_RESPONSE_SCREEN_STATUS_PENDING_GEOMETRY_RULE,
        "data_free": True,
        "local_preparation_only": True,
        "measured_fit_release": False,
        "archive_opened": False,
        "pace_jobs": 0,
        "manager_touched": False,
        "deployment": "none",
        "protocol": payload,
        "result_disclosure": "contract only; no measured archive, response payload, fit, registration, or geometry decision",
    }


def _load_actual_shards(archive_root: str | Path) -> tuple[Any, ...]:
    """Load only the declared local archive layout at explicit CLI execution."""

    root = Path(archive_root).expanduser().resolve()
    shard_root = root / "converted_v3_joint8_fullpol" / "shards"
    paths = tuple(shard_root / f"pass{pass_id}_hh.npz" for pass_id in LOC.MEASURED_RESPONSE_SCREEN_PASSES)
    return tuple(
        ACQ.load_native_shard(
            path,
            expected_pass_id=pass_id,
            expected_polarization="hh",
            expected_scene_id=LOC.MEASURED_RESPONSE_SCREEN_SCENE,
        )
        for pass_id, path in zip(LOC.MEASURED_RESPONSE_SCREEN_PASSES, paths)
    )


def _write_outputs(output_dir: str | Path, result: dict[str, Any]) -> None:
    """Write future-run maps/reports without exporting raw responses."""

    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    dataset = result["dataset"]
    grid = result["grid"]
    maps = result["maps"]
    np.savez_compressed(
        out / "stage_a_maps.npz",
        q_xyz_m=np.asarray(grid.points_xyz_m),
        normalized_power_by_panel=np.asarray(maps.normalized_power_by_panel),
        matched_complex_by_panel=np.asarray(maps.matched_complex_by_panel),
        noncoherent_rank_summary=np.asarray(maps.noncoherent_rank_summary),
    )
    stage_b = result["stage_b"]
    if stage_b.maps is not None and stage_b.grid is not None:
        np.savez_compressed(
            out / "stage_b_maps.npz",
            q_xyz_m=np.asarray(stage_b.grid.points_xyz_m),
            normalized_power_by_panel=np.asarray(stage_b.maps.normalized_power_by_panel),
            matched_complex_by_panel=np.asarray(stage_b.maps.matched_complex_by_panel),
            noncoherent_rank_summary=np.asarray(stage_b.maps.noncoherent_rank_summary),
        )
    psf = result.get("psf")
    if psf is not None:
        np.savez_compressed(
            out / "psf_diagnostics.npz",
            reference_xyz_m=np.asarray(psf.reference_xyz_m),
            normalized_power_by_panel=np.asarray(psf.normalized_power_by_panel),
            discovery_rank_summary=np.asarray(psf.discovery_rank_summary),
            confirmation_rank_summary=np.asarray(psf.confirmation_rank_summary),
        )
    offgrid = result.get("offgrid")
    if offgrid is not None:
        np.savez_compressed(
            out / "offgrid_sensitivity.npz",
            reference_points_xyz_m=np.asarray(offgrid.reference_points_xyz_m),
            panel_scores=np.asarray(offgrid.panel_scores),
            normalized_power_by_panel_corner_candidate=np.asarray(offgrid.normalized_power_by_panel_corner_candidate),
        )
    bridge = result.get("bridge")
    if bridge is not None:
        np.savez_compressed(
            out / "raw_source_bridge.npz",
            raw_equivalent_scores=np.asarray(bridge.raw_equivalent_scores),
            source_scores=np.asarray(bridge.source_scores),
        )
    (out / "preflight.json").write_text(json.dumps(dataset.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
    (out / "readiness_report.json").write_text(json.dumps(result["public_result"], indent=2, sort_keys=True), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    figure, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    all_images = [_power_db(np.asarray(maps.normalized_power_by_panel[index]).reshape(7, 7, 6).max(axis=2)) for index in range(8)]
    vmin = float(min(np.min(image) for image in all_images))
    vmax = float(max(np.max(image) for image in all_images))
    if vmax <= vmin:
        vmax = vmin + 1.0
    image_artist = None
    for index, axis in enumerate(axes.flat):
        image_artist = axis.imshow(all_images[index].T, origin="lower", aspect="equal", vmin=vmin, vmax=vmax)
        axis.set_title(maps.panels[index].panel_id)
        axis.set_xlabel("q x sample")
        axis.set_ylabel("q y sample")
    figure.colorbar(image_artist, ax=axes.ravel().tolist(), label=f"normalized power (dB; floor {POWER_DISPLAY_FLOOR_DB:g} dB)")
    figure.savefig(out / "stage_a_common_scale_mip.png", dpi=140)
    plt.close(figure)
    if stage_b.maps is not None:
        figure, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
        fine_images = [_power_db(np.asarray(stage_b.maps.normalized_power_by_panel[index]).reshape(8, 8, 8).max(axis=2)) for index in range(8)]
        fine_vmin = float(min(np.min(image) for image in fine_images))
        fine_vmax = float(max(np.max(image) for image in fine_images))
        if fine_vmax <= fine_vmin:
            fine_vmax = fine_vmin + 1.0
        fine_artist = None
        for index, axis in enumerate(axes.flat):
            fine_artist = axis.imshow(fine_images[index].T, origin="lower", aspect="equal", vmin=fine_vmin, vmax=fine_vmax)
            axis.set_title(f"fine {stage_b.maps.panels[index].panel_id}")
        figure.colorbar(fine_artist, ax=axes.ravel().tolist(), label=f"normalized power (dB; floor {POWER_DISPLAY_FLOOR_DB:g} dB)")
        figure.savefig(out / "stage_b_common_scale_mip.png", dpi=140)
        plt.close(figure)
    if psf is not None:
        figure, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
        for index, axis in enumerate(axes.flat):
            profile = np.asarray(psf.normalized_power_by_panel[index], dtype=np.float64)
            axis.plot(np.arange(profile.size), profile)
            axis.axvline(294 - 0.5, color="black", linestyle="--", linewidth=0.8, label="coarse/fine boundary")
            axis.set_title(f"PSF {maps.panels[index].panel_id}")
            axis.set_xlabel("coarse+fine union q index")
            axis.set_ylabel("normalized power")
            axis.legend(fontsize=7, loc="best")
        figure.savefig(out / "panel_psf_profiles.png", dpi=140)
        plt.close(figure)
    if offgrid is not None:
        figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
        image = np.asarray(offgrid.panel_scores)
        axis.imshow(image, aspect="auto", interpolation="nearest", vmin=0.0, vmax=1.0)
        axis.set_title("half-cell off-grid sensitivity (diagnostic only)")
        axis.set_xlabel("corner")
        axis.set_ylabel("panel")
        figure.savefig(out / "offgrid_sensitivity.png", dpi=140)
        plt.close(figure)


def execute(shards: Sequence[Any], *, output_dir: str | Path | None = None) -> dict[str, Any]:
    """Execute the actual measured binding after an explicit CLI archive load."""

    dataset = LOC.prepare_tophat_measured_response_screen_dataset(tuple(shards))
    grid = LOC.build_tophat_measured_response_screen_grid()
    maps = LOC.stream_tophat_measured_response_screen_maps(dataset, grid)
    stage_a = LOC.evaluate_tophat_measured_response_stage_a(dataset, grid, maps)
    stage_b = LOC.run_tophat_measured_response_stage_b(dataset, stage_a)
    psf = offgrid = bridge = None
    reference_index = reference_xyz = None
    if stage_b.grid is not None:
        reference_index, reference_xyz = LOC.choose_tophat_measured_screen_reference(grid, maps)
        psf = LOC.stream_tophat_measured_response_psfs(dataset, grid, stage_b.grid, reference_xyz)
        offgrid = LOC.stream_tophat_measured_response_offgrid_sensitivity(dataset, reference_xyz)
        bridge = LOC.stream_tophat_measured_response_raw_source_bridge(dataset)
    ledger = LOC.TophatMeasuredResponseScreenWorkLedger()
    executed_categories = {"stage_a_coarse_maps": maps.kernel_sample_terms}
    if stage_b.maps is not None:
        executed_categories["stage_b_fine_maps"] = stage_b.maps.kernel_sample_terms
    if psf is not None:
        executed_categories["union_forward_and_panel_psf"] = psf.kernel_sample_terms
    if offgrid is not None:
        executed_categories["offgrid_reference_sensitivity"] = offgrid.kernel_sample_terms
    if bridge is not None:
        executed_categories["raw_source_bridge"] = bridge.kernel_sample_terms
    executed_terms = int(sum(executed_categories.values()))
    public = {
        "status": stage_a.status,
        "data_free": False,
        "local_preparation_only": False,
        "measured_fit_release": False,
        "archive_opened": True,
        "future_shard_binding_supplied": True,
        "pace_jobs": 0,
        "manager_touched": False,
        "deployment": "none",
        "dataset": dataset.as_dict(),
        "grid": grid.as_dict(),
        "maps": maps.as_dict(),
        "stage_a": stage_a.as_dict(),
        "stage_b": stage_b.as_dict(),
        "deterministic_reference": None if reference_xyz is None else {
            "coarse_grid_index": reference_index,
            "xyz_m": reference_xyz.tolist(),
            "diagnostic_only": True,
        },
        "psf": None if psf is None else psf.as_dict(),
        "offgrid_sensitivity": None if offgrid is None else offgrid.as_dict(),
        "raw_source_bridge": None if bridge is None else bridge.as_dict(),
        "executed_work": {
            "sample_term_definition": "executed direct kernel/sample terms; not exact FLOPs",
            "categories": executed_categories,
            "total_direct_kernel_sample_terms": executed_terms,
            "planned_total_direct_kernel_sample_terms": ledger.total_direct_kernel_sample_terms,
            "diagnostics_skipped_without_inspection": stage_b.grid is None,
        },
        "ledger": ledger.as_dict(),
        "noncoherent_rank_summary": {
            "candidate_count": int(grid.points_xyz_m.shape[0]),
            "summary_only": True,
            "array_serialized": False,
        },
        "qualification": {
            "status": stage_a.status,
            "accepted_native_R_t": None,
            "ground_height": "blocked",
            "working_transform": None if stage_a.selected_cube_center_xyz_m is None else {
                "R": "I",
                "t_working": stage_a.selected_cube_center_xyz_m.tolist(),
                "semantics": "conditional response-informed operational ROI only",
            },
            "t_working": None if stage_a.selected_cube_center_xyz_m is None else stage_a.selected_cube_center_xyz_m.tolist(),
            "inspection_center": None if stage_a.inspection_cube_center_xyz_m is None else stage_a.inspection_cube_center_xyz_m.tolist(),
            "inspection_label": stage_a.inspection_label,
            "context_dominated": stage_a.context_dominated,
            "context_dominated_rule": "descriptive flag iff all 8 panels have outside_max/inside_max >= 1; not a qualification gate",
        },
        "raw_response_serialized": False,
        "source_af_exact_once": True,
        "no_synthetic_far_side_model": True,
        "no_dense_H_cache": True,
    }
    result = {
        "public_result": public,
        "dataset": dataset,
        "grid": grid,
        "maps": maps,
        "stage_b": stage_b,
        "ledger": ledger,
        "psf": psf,
        "offgrid": offgrid,
        "bridge": bridge,
    }
    if output_dir is not None:
        _write_outputs(output_dir, result)
        public["output_dir"] = str(Path(output_dir).expanduser().resolve())
        public["outputs_written"] = True
    return result


def run(archive_root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Execute against an explicit trusted local archive root and output directory."""

    return execute(_load_actual_shards(archive_root), output_dir=output_dir)["public_result"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", required=True, help="local root containing converted_v3_joint8_fullpol/shards")
    parser.add_argument("--output-dir", required=True, help="directory for map arrays and truth-free reports")
    args = parser.parse_args()
    print(json.dumps(run(args.archive_root, args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
