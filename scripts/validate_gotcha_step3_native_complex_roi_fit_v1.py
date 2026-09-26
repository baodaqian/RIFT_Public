"""Validate the local fake-record native-complex directional-SH ROI control."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_roi():
    name = "gotcha_native_complex_roi_fit"
    if name in sys.modules:
        return sys.modules[name]
    path = ROOT / "rift" / "gotcha_native_complex_roi_fit.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ROI = _load_roi()
PROTOCOL = ROOT / "protocols" / "gotcha_step3_native_complex_roi_fit_v1.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate only the local fake-record ROI control.")
    parser.parse_args(argv)
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    expected = ROI.protocol_payload()
    for key in (
        "schema",
        "target_id",
        "source_af_formula",
        "global_complex_gain",
        "fit_release_status",
        "count_contracts",
        "actual_mode",
        "placements",
        "cgls_contract",
        "range_subspace_contract",
    ):
        if payload.get(key) != expected.get(key):
            raise AssertionError(f"protocol field {key} drifted")
    actual_mode = payload["actual_mode"]
    if actual_mode["solver_default"] != "gd" or actual_mode["solver_choices"] != ["gd", "cgls"]:
        raise AssertionError("actual solver choice contract drifted")
    if actual_mode["cgls_max_iterations"] != ROI.MAX_CGLS_ITERATIONS:
        raise AssertionError("CGLS maximum iteration contract drifted")
    if actual_mode["cgls_diagnostic_steps"] != list(ROI.CGLS_DIAGNOSTIC_STEPS):
        raise AssertionError("CGLS diagnostic checkpoint contract drifted")
    if actual_mode["range_comparison_mode"] != "actual_range_subspace_comparison" or not actual_mode[
        "range_comparison_requires_saved_native_root"
    ]:
        raise AssertionError("range comparison CLI contract drifted")
    cgls_contract = payload["cgls_contract"]
    if not cgls_contract["available_only_in_actual_mode"] or not cgls_contract["train_records_only"]:
        raise AssertionError("CGLS must remain actual-mode TRAIN-only")
    if cgls_contract["initialization"] != "zero":
        raise AssertionError("CGLS must start from zero coefficients")
    if not cgls_contract["no_extra_factor_two"] or not cgls_contract["no_denominator_floor"]:
        raise AssertionError("CGLS normalization/denominator contract drifted")
    if "not a zero-residual test" not in cgls_contract["stationarity_test"]:
        raise AssertionError("CGLS stationarity must be distinguished from zero residual")
    range_contract = payload["range_subspace_contract"]
    if range_contract["mode"] != "actual_range_subspace_comparison":
        raise AssertionError("range comparison mode contract drifted")
    if range_contract["point_response"]["samples"] != ROI.RANGE_SUBSPACE_SAMPLES:
        raise AssertionError("range point-response sample count drifted")
    if range_contract["point_response"]["golden_section_iterations"] != ROI.RANGE_SUBSPACE_GOLDEN_ITERATIONS:
        raise AssertionError("range golden-section iteration count drifted")
    if range_contract["range_grid"]["guard"] != "fixed 2g only" or not range_contract["range_grid"]["require_M_lt_K"]:
        raise AssertionError("range fixed guard/grid contract drifted")
    if not range_contract["dense_T_or_P_materialized"] is False:
        raise AssertionError("range comparison must not materialize T or P")
    if range_contract["h"] != "g exactly; g/2 rejected for scored mode":
        raise AssertionError("range spacing contract drifted")
    if range_contract["arms"] != [
        "saved_native_objective_cgls24",
        "range_focused_isotropic_bp",
        "range_focused_degree1_cgls24",
    ]:
        raise AssertionError("range comparison arms drifted")
    if payload["test_payload_opened"] or payload["actual_archive_opened"] or payload["pace_action"]:
        raise AssertionError("local validator must remain archive/PACE-free")
    if payload["field_model"]["all_voxels_active"] is not True or payload["field_model"]["adaptive_prune_grow_support"]:
        raise AssertionError("support must be fixed and all-active")
    if payload["schedule"]["minimum_actual_optimizer_updates"] < 12:
        raise AssertionError("non-smoke update schedule is too short")
    if payload["schedule"]["validation_use"] != "diagnostic_only_fixed_points":
        raise AssertionError("validation must be diagnostic-only")
    if payload["schedule"]["selected_model"] != "update_12_final_for_train_and_validation":
        raise AssertionError("update-12 final state must be the sole selected model")
    if "checkpoint_best" in payload["persistence"]["full_state"]:
        raise AssertionError("validation-selected best checkpoint must not be declared")
    if payload["schedule"]["atomic_per_update"] or payload["schedule"]["resumable_per_update"]:
        raise AssertionError("fixed-point control must not claim atomic/resumable per-update persistence")
    model, train, validation, planted = ROI.make_fake_camry_control_case(
        train_count=4, validation_count=2, frequency_count=8
    )
    ROI.validate_record_set(train, split="train")
    ROI.validate_record_set(validation, split="validation")
    if model.point_count != 1000 or model.coefficient_shape != (1000, 4):
        raise AssertionError("Camry support is not fixed 10^3 degree-1 SH")
    readout = model.readout(planted)
    if tuple(readout["shape"]) != (100, 100, 100) or readout["point_count"] != 1_000_000:
        raise AssertionError("Camry readout is not the exact 100^3 half-open grid")
    if readout["label"] != ROI.READOUT_LABEL:
        raise AssertionError("readout lacks coarse-support disclosure")
    result = ROI.fit_fixed_schedule(model, train, validation, config=ROI.FitConfig())
    if len(result.history) != 4 or not np.isfinite(
        np.asarray([[entry["train_loss"], entry["validation_loss"]] for entry in result.history])
    ).all():
        raise AssertionError("fixed schedule history is not finite at 0/4/8/12")
    if not np.array_equal(result.coefficients_selected, result.coefficients_final):
        raise AssertionError("selected model is not the update-12 final state")
    if any(entry["validation_use"] != "diagnostic_only_fixed_point" or entry["selected_checkpoint"] for entry in result.history):
        raise AssertionError("validation was used to select a checkpoint")
    tophat_model, tophat_train, tophat_planted = ROI.make_fake_tophat_control_case(
        records_per_panel=1, frequency_count=4
    )
    ROI.validate_record_set(tophat_train, split="train", target_id="tophat")
    if len(tophat_train) != 8 or tophat_model.point_count != 512:
        raise AssertionError("TopHat fake P1/P7 panel binding is incomplete")
    panel_keys = {(record.identity.pass_id, record.identity.sector_id) for record in tophat_train}
    if panel_keys != {(pass_id, sector_id) for pass_id in (1, 7) for sector_id in (2, 92, 182, 272)}:
        raise AssertionError("TopHat panel identities drifted")
    tophat_prediction = tophat_model.forward(tophat_train, tophat_planted)
    if len(tophat_prediction.values) != 8:
        raise AssertionError("TopHat panels were pooled instead of retained as ragged records")
    tophat_readout = tophat_model.readout(tophat_planted)
    if tuple(tophat_readout["shape"]) != (40, 40, 40) or tophat_readout["mid_z_index_for_local_zero"] != 20:
        raise AssertionError("TopHat exact readout or geometric z=0 slice drifted")
    tophat_validation_base = ROI.make_fake_tophat_source_af_records(
        role="validation", records_per_panel=1, frequency_count=4
    )
    tophat_validation_target = tophat_model.forward(tophat_validation_base, tophat_planted)
    tophat_validation = tuple(
        record.with_effective_response(value)
        for record, value in zip(tophat_validation_base, tophat_validation_target.values)
    )
    ROI.validate_record_set(tophat_validation, split="validation", target_id="tophat")
    tophat_result = ROI.fit_full_aperture_schedule(
        tophat_model,
        tophat_train,
        tophat_validation,
        config=ROI.actual_fit_config(tophat_model),
    )
    if tophat_result.history[-1]["step"] != 12 or tophat_result.history[-1]["validation_loss"] is None:
        raise AssertionError("TopHat fixed validation selector was not used diagnostically")
    print(
        json.dumps(
            {
                "schema": ROI.SCHEMA,
                "status": "PASS_LOCAL_FAKE_RECORDS_ONLY",
                "train_records": len(train),
                "validation_records": len(validation),
                "support_points": model.point_count,
                "readout_points": readout["point_count"],
                "actual_optimizer_updates": 12,
                "history_points": len(result.history),
                "archive_opened": False,
                "pace_action": False,
                "manager_touched": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
