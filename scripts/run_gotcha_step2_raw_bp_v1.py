"""Run the reviewed GOTCHA Step-2 P1 raw-BP profile diagnostic.

This driver is intentionally NumPy-first and loads the Batch-A and Step-2
modules by path.  It does not import the RIFT package, fit a calibration, open
test data, choose an ROI, or chain into a later stage.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import importlib.util
import json
from pathlib import Path
import sys
import textwrap
import time

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ACQ = _load_module("gotcha_acquisition_for_step2_raw_bp_v1", PROJECT_ROOT / "rift" / "gotcha_acquisition.py")
CTL = _load_module("gotcha_step2_controls_for_raw_bp_v1", PROJECT_ROOT / "rift" / "gotcha_step2_controls.py")


PROTOCOL_SCHEMA = "rift_gotcha_step2_p1_raw_bp_profile_protocol_v1"
DRIVER_SCHEMA = "rift_gotcha_step2_p1_raw_bp_profile_driver_v1"


def _json_ready(value):
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_json_ready(value), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _read_protocol(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("protocol must be a JSON object")
    return value


def _assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise ValueError(f"protocol {label} must be {expected!r}; got {actual!r}")


def _validate_protocol(protocol: Mapping[str, object], stage: str) -> None:
    _assert_equal(stage, "profile", "CLI stage")
    _assert_equal(protocol.get("schema"), PROTOCOL_SCHEMA, "schema")
    _assert_equal(protocol.get("stage"), "profile", "stage")

    source = protocol.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("protocol source must be an object")
    _assert_equal(source.get("archive_root_contract"), "converted_v3_joint8_fullpol", "source.archive_root_contract")
    _assert_equal(source.get("relative_path"), "shards/pass1_hh.npz", "source.relative_path")
    _assert_equal(source.get("pass_id"), 1, "source.pass_id")
    _assert_equal(source.get("polarization"), "hh", "source.polarization")

    selection = protocol.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("protocol selection must be an object")
    for key, expected in {
        "role": "train",
        "sector_id": 2,
        "canonical_identity": "(pass, polarization, sector, pulse)",
        "order": "ascending canonical identity",
        "all_native_pulses_in_sector": True,
    }.items():
        _assert_equal(selection.get(key), expected, f"selection.{key}")

    native = protocol.get("native_contract")
    if not isinstance(native, Mapping):
        raise ValueError("protocol native_contract must be an object")
    for key, expected in {
        "frequency_policy": "native_stored_exact",
        "r0_field": "r0",
        "r0_unit": "m",
        "geometry": "paired_monostatic_tx_equals_rx_same_observation",
        "phase_hypothesis": CTL.PHASE_HYPOTHESIS_NAME,
        "phase_forward": CTL.PHASE_HYPOTHESIS_FORWARD,
        "phase_adjoint": CTL.PHASE_HYPOTHESIS_ADJOINT,
        "phase_status": CTL.PHASE_HYPOTHESIS_STATUS,
        "autofocus": "raw_unapplied_channel_owned",
        "autofocus_application": "not_run_sign_order_and_units_unverified",
        "amplitude_weighting": False,
        "normalization": "mean_native_sample",
        "uniform_frequency_grid": False,
    }.items():
        _assert_equal(native.get(key), expected, f"native_contract.{key}")

    support = protocol.get("support")
    if not isinstance(support, Mapping):
        raise ValueError("protocol support must be an object")
    _assert_equal(support.get("schema"), CTL.H0_SUPPORT_SCHEMA, "support.schema")
    _assert_equal(support.get("support_mode"), "plane_no_height", "support.support_mode")
    _assert_equal(support.get("bounds_m"), {"x": [-8.0, 8.0], "y": [-8.0, 8.0], "z": [0.0, 0.0]}, "support.bounds_m")
    _assert_equal(support.get("sampling"), {"spacing_m": [0.25, 0.25, 1.0], "shape": [65, 65, 1]}, "support.sampling")
    for key, expected in {
        "frame_contract": "antenna_xyz_unchanged",
        "registration_status": "unresolved",
        "support_status": CTL.SUPPORT_STATUS,
        "runtime_profile_origin_only": True,
        "target_roi_or_localization_claim": False,
        "max_kernel_evaluations": 250_000_000,
        "point_chunk_size": 4096,
    }.items():
        _assert_equal(support.get(key), expected, f"support.{key}")
    _assert_equal(support.get("units"), {"x": "m", "y": "m", "z": "m", "r0": "m", "frequency": "Hz"}, "support.units")
    _assert_equal(support.get("phase_hypothesis"), CTL.PHASE_HYPOTHESIS_NAME, "support.phase_hypothesis")
    _assert_equal(support.get("autofocus_status"), "raw_unapplied_channel_owned", "support.autofocus_status")

    future = protocol.get("future_path")
    if not isinstance(future, Mapping):
        raise ValueError("protocol future_path must be an object")
    for key in ("coarse", "refinement"):
        text = str(future.get(key, ""))
        if not text.startswith("NOT RUN"):
            raise ValueError(f"future_path.{key} must remain explicitly NOT RUN")
    _assert_equal(future.get("auto_chain"), False, "future_path.auto_chain")

    outputs = protocol.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("protocol outputs must be an object")
    _assert_equal(
        outputs,
        {
            "raw_complex_bp": "bp_complex.npy",
            "x_coordinates": "x.npy",
            "y_coordinates": "y.npy",
            "plot": "bp_native_xy_z0.png",
            "plot_display": {
                "quantity": "20log10(abs(BP)/A_ref)",
                "A_ref": "profile peak magnitude",
                "clip_db": [-40.0, 0.0],
                "x_horizontal": True,
                "y_vertical": True,
                "origin": "lower",
                "aspect": "equal",
            },
        },
        "outputs",
    )


def _id_key(identity) -> tuple[int, str, int, int]:
    return (
        int(identity.pass_id),
        str(identity.polarization).lower(),
        int(identity.sector_id),
        int(identity.pulse_index),
    )


def _support_declaration(protocol: Mapping[str, object]) -> dict:
    support = dict(protocol["support"])
    return support


def _select_train_sector_observations(shard, protocol: Mapping[str, object]):
    selection = protocol["selection"]
    identities = tuple(shard.observation_ids)
    roles = tuple(str(value).lower() for value in np.asarray(shard.role).tolist())
    if len(identities) != len(roles):
        raise ValueError("native identity and role headers have different lengths")
    role_by_id = dict(zip(identities, roles))
    loaded_roles = sorted(set(roles))
    if loaded_roles != ["train", "validation"]:
        raise ValueError(f"loaded non-test response roles must be train+validation; got {loaded_roles}")
    if any(role == "test" for role in roles):
        raise ValueError("test payload rows must remain sealed")
    selected = tuple(
        sorted(
            (
                identity
                for identity in identities
                if role_by_id[identity] == "train"
                and int(identity.sector_id) == int(selection["sector_id"])
            ),
            key=_id_key,
        )
    )
    if not selected:
        raise ValueError("protocol sector has no train observations")
    if tuple(sorted(selected, key=_id_key)) != selected:
        raise ValueError("selected identities are not in canonical order")
    for identity in selected:
        if _id_key(identity)[:3] != (1, "hh", 2):
            raise ValueError(f"selected identity violates protocol: {identity}")
    # This is the one and only payload materialization call, and it is made
    # with the complete, canonical, train-only identity list.
    observations = tuple(shard.observations(selected))
    if tuple(observation.identity for observation in observations) != selected:
        raise ValueError("materialized observations do not preserve selected canonical IDs")
    for observation in observations:
        identity = observation.identity
        if str(observation.role).lower() != "train" or int(identity.sector_id) != 2:
            raise ValueError("selected observation role/sector mismatch")
        if int(identity.pass_id) != 1 or str(identity.polarization).lower() != "hh":
            raise ValueError("selected observation pass/polarization mismatch")
    return selected, observations, loaded_roles


def _validate_native_contract(shard, observations, protocol: Mapping[str, object]) -> None:
    native = protocol["native_contract"]
    if shard.phase_reference.frequency_values != native["frequency_policy"]:
        raise ValueError("native frequency provenance does not match protocol")
    if shard.phase_reference.reference_range_field != native["r0_field"]:
        raise ValueError("native r0 field does not match protocol")
    if shard.phase_reference.geometry_contract != native["geometry"]:
        raise ValueError("native geometry contract does not match protocol")
    if shard.autofocus.mode != ACQ.AUTOFOCUS_RAW or shard.autofocus.applied:
        raise ValueError("HH profile requires raw/unapplied autofocus provenance")
    if shard.autofocus.official_available is not True:
        raise ValueError("HH profile requires channel-owned raw autofocus arrays")
    for observation in observations:
        if not np.array_equal(np.asarray(observation.frequencies_hz), np.asarray(shard.frequencies_hz)):
            raise ValueError("selected observation frequency vector was not preserved exactly")
        if observation.phase_reference.reference_range_field != "r0":
            raise ValueError("selected observation lost native per-pulse r0 provenance")
        if observation.autofocus.applied:
            raise ValueError("selected observation has applied autofocus")


def _process_rss_bytes() -> int | None:
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value * 1024 if sys.platform != "darwin" else value
    except Exception:
        return None


def _profile_title(selected_count: int, frequency_counts: list[int]) -> str:
    total_samples = int(sum(frequency_counts))
    minimum = int(min(frequency_counts))
    maximum = int(max(frequency_counts))
    return (
        "P1 HH sector2 conditional raw BP profile | "
        f"selected pulses={selected_count}, total native samples={total_samples}, F/observation={minimum}-{maximum} | "
        "native per-pulse r0 | raw/unapplied AF | Demanet phase candidate | "
        "unweighted mean_native_sample | z_native=0 numerical plane | registration unresolved"
    )


def _require_pillow() -> None:
    try:
        import PIL  # noqa: F401
    except Exception as error:
        raise RuntimeError("Pillow is required for the deterministic profile PNG renderer") from error


def _write_profile_png(path: Path, values: np.ndarray, support, title: str) -> dict[str, object]:
    grid = np.asarray(values, dtype=np.complex128).reshape(tuple(support.grid_shape), order="C")
    magnitude = np.abs(grid[:, :, 0])
    reference = float(np.max(magnitude))
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError("profile BP peak magnitude must be positive and finite")
    db = 20.0 * np.log10(np.maximum(magnitude / reference, 10.0 ** (-100.0 / 20.0)))
    clipped = np.clip(db, -40.0, 0.0)
    normalized = (clipped + 40.0) / 40.0
    # Native grid is [x, y, z] with x-major/y-next/z-fastest C order.  The
    # transpose makes x horizontal and y vertical; Pillow's top-left raster
    # is flipped once so the displayed origin is native lower-left.
    display = normalized.T[::-1, :]
    color = np.empty((*display.shape, 3), dtype=np.uint8)
    color[:, :, 0] = np.asarray(np.clip(255.0 * display, 0.0, 255.0), dtype=np.uint8)
    color[:, :, 1] = np.asarray(np.clip(255.0 * (1.0 - np.abs(2.0 * display - 1.0)), 0.0, 255.0), dtype=np.uint8)
    color[:, :, 2] = np.asarray(np.clip(255.0 * (1.0 - display), 0.0, 255.0), dtype=np.uint8)

    from PIL import Image, ImageDraw, ImageFont
    from PIL.PngImagePlugin import PngInfo

    font = ImageFont.load_default()
    scale = 4
    color_image = Image.fromarray(color, mode="RGB").resize(
        (color.shape[1] * scale, color.shape[0] * scale),
        resample=Image.Resampling.NEAREST,
    )
    image_width, image_height = color_image.size
    header_lines = []
    header_lines.extend(textwrap.wrap(title, width=86, break_long_words=False, break_on_hyphens=False))
    header_lines.extend(
        [
            "display: 20log10(|BP|/A_ref), A_ref = profile peak magnitude, clip = [-40,0] dB",
            "phase: exp(-i 4*pi*f/c (R-r0)); reference: native per-pulse r0; AF: raw/unapplied",
            "conditional raw-BP diagnostic; unweighted mean_native_sample; z_native=0 numerical plane",
            f"native x=[{support.bounds_m['x'][0]:g},{support.bounds_m['x'][1]:g}] m, y=[{support.bounds_m['y'][0]:g},{support.bounds_m['y'][1]:g}] m; ticks shown; origin=lower; aspect=equal",
            "registration unresolved; not calibrated, registered, height, localization, or geometry evidence",
        ]
    )
    line_height = 12
    header_height = 8 + line_height * len(header_lines)
    left, top = 72, header_height
    right = 120
    bottom = 72
    canvas = Image.new("RGB", (left + image_width + right, top + image_height + bottom), "white")
    canvas.paste(color_image, (left, top))
    draw = ImageDraw.Draw(canvas)
    for index, line in enumerate(header_lines):
        draw.text((5, 5 + index * line_height), line, fill="black", font=font)
    draw.rectangle((left, top, left + image_width - 1, top + image_height - 1), outline="black")
    draw.text((left + image_width // 2 - 20, top + image_height + 32), "x (m)", fill="black", font=font)
    draw.text((4, top + image_height // 2), "y (m)", fill="black", font=font)
    x_bounds = support.bounds_m["x"]
    y_bounds = support.bounds_m["y"]
    x_ticks = (x_bounds[0], (x_bounds[0] + x_bounds[1]) / 2.0, x_bounds[1])
    y_ticks = (y_bounds[0], (y_bounds[0] + y_bounds[1]) / 2.0, y_bounds[1])
    for fraction, value in zip((0.0, 0.5, 1.0), x_ticks):
        x = int(round(left + fraction * (image_width - 1)))
        draw.line((x, top + image_height, x, top + image_height + 5), fill="black")
        draw.text((x - 12, top + image_height + 8), f"{value:g}", fill="black", font=font)
    for fraction, value in zip((1.0, 0.5, 0.0), y_ticks):
        y = int(round(top + fraction * (image_height - 1)))
        draw.line((left - 5, y, left, y), fill="black")
        draw.text((5, y - 4), f"{value:g}", fill="black", font=font)
    bar_x, bar_y, bar_w, bar_h = left + image_width + 18, top, 15, image_height
    bar = np.linspace(1.0, 0.0, bar_h)[:, None]
    bar_rgb = np.empty((bar_h, bar_w, 3), dtype=np.uint8)
    bar_rgb[:, :, 0] = np.asarray(255.0 * bar, dtype=np.uint8)
    bar_rgb[:, :, 1] = np.asarray(255.0 * (1.0 - np.abs(2.0 * bar - 1.0)), dtype=np.uint8)
    bar_rgb[:, :, 2] = np.asarray(255.0 * (1.0 - bar), dtype=np.uint8)
    canvas.paste(Image.fromarray(bar_rgb, mode="RGB"), (bar_x, bar_y))
    for tick, label in ((0.0, "0"), (0.5, "-20"), (1.0, "-40")):
        y = int(round(bar_y + tick * (bar_h - 1)))
        draw.line((bar_x + bar_w, y, bar_x + bar_w + 4, y), fill="black")
        draw.text((bar_x + bar_w + 7, y - 4), label, fill="black", font=font)
    draw.text((bar_x - 1, bar_y + bar_h + 7), "dB", fill="black", font=font)
    png_info = PngInfo()
    png_info.add_text("Title", title)
    png_info.add_text("Description", "\n".join(header_lines))
    png_info.add_text("NativeGrid", "x horizontal; y vertical; transpose then vertical flip for origin lower; equal native metre axes")
    canvas.save(path, format="PNG", pnginfo=png_info)
    return {
        "backend": "Pillow_dependency_light",
        "display_quantity": "20log10(abs(BP)/A_ref)",
        "A_ref": reference,
        "db_clip": [-40.0, 0.0],
        "grid_order": "[x,y,z] C-order, x-major/y-next/z-fastest",
        "transpose_for_display": True,
        "raster_vertical_flip_for_origin_lower": True,
        "x_horizontal": True,
        "y_vertical": True,
        "origin": "lower",
        "aspect": "equal",
        "x_bounds_m": list(support.bounds_m["x"]),
        "y_bounds_m": list(support.bounds_m["y"]),
        "z_native_m": 0.0,
        "colorbar_ticks_db": [0.0, -20.0, -40.0],
        "header_lines": header_lines,
        "native_x_ticks_m": list(x_ticks),
        "native_y_ticks_m": list(y_ticks),
        "pixel_scale": scale,
    }


def run_profile(protocol_path: str | Path, archive_root: str | Path, output_dir: str | Path, stage: str = "profile") -> dict[str, object]:
    protocol_path = Path(protocol_path)
    archive_root = Path(archive_root)
    output_dir = Path(output_dir)
    protocol = _read_protocol(protocol_path)
    _validate_protocol(protocol, stage)
    _require_pillow()
    if output_dir.exists():
        raise FileExistsError(f"output directory must be fresh: {output_dir}")
    relative = Path(str(protocol["source"]["relative_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("protocol source path must be a safe relative path")
    archive_path = archive_root / relative
    if not archive_path.is_file():
        raise FileNotFoundError(f"protocol archive is missing: {archive_path}")

    shard = ACQ.load_native_shard(
        archive_path,
        expected_pass_id=1,
        expected_polarization="hh",
        expected_scene_id="gotcha_v1_joint8_fullpol",
    )
    selected_ids, observations, loaded_roles = _select_train_sector_observations(shard, protocol)
    _validate_native_contract(shard, observations, protocol)
    if shard.metadata.get("test_opened") is not False or shard.metadata.get("test_payload_included") is not False:
        raise ValueError("Gate-1 metadata does not prove that test payload remained sealed")

    support_decl = _support_declaration(protocol)
    point_chunk_size = int(protocol["support"]["point_chunk_size"])
    support = CTL.NativeFrameH0Support.from_mapping(support_decl)
    preflight = CTL.preflight_support_declaration(support_decl, observations, point_count=support.point_count)
    preflight["resource_estimate"]["point_chunk_size"] = point_chunk_size
    preflight["resource_estimate"]["point_chunks_per_observation"] = int(
        (support.point_count + point_chunk_size - 1) // point_chunk_size
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    bp = CTL.conditional_backproject(
        support_decl,
        observations,
        normalization=protocol["native_contract"]["normalization"],
        point_chunk_size=point_chunk_size,
    )
    elapsed = time.perf_counter() - started
    raw_bp = np.asarray(bp.values, dtype=np.complex128)
    x = np.linspace(support.bounds_m["x"][0], support.bounds_m["x"][1], support.grid_shape[0], dtype=np.float64)
    y = np.linspace(support.bounds_m["y"][0], support.bounds_m["y"][1], support.grid_shape[1], dtype=np.float64)
    np.save(output_dir / "bp_complex.npy", raw_bp)
    np.save(output_dir / "x.npy", x)
    np.save(output_dir / "y.npy", y)

    frequency_counts = [int(np.asarray(observation.frequencies_hz).size) for observation in observations]
    title = _profile_title(len(observations), frequency_counts)
    plot_metadata = _write_profile_png(output_dir / "bp_native_xy_z0.png", raw_bp, support, title)
    disclosure = {
        "loaded_response_roles": loaded_roles,
        "used_response_roles": ["train"],
        "loaded_response_count": int(shard.view_count),
        "selected_response_count": len(observations),
        "selected_ids": [identity.as_dict() for identity in selected_ids],
        "selected_r0_m": [
            {"id": observation.identity.as_dict(), "r0_m": float(observation.r0_m)}
            for observation in observations
        ],
        "selected_frequency_counts": frequency_counts,
        "archive_validation": {
            "performed_by_batch_a_loader": True,
            "scope": "loaded non-test train+validation archive headers and response schema/finiteness",
            "response_derived_science_used": False,
            "used_for_bp": False,
        },
        "test_payload_opened": False,
        "validation_used_for_bp": False,
        "validation_used_for_scale_energy_peak_or_refinement": False,
    }
    _write_json(output_dir / "protocol_echo.json", protocol)
    _write_json(output_dir / "preflight.json", {"preflight": preflight, "disclosure": disclosure})
    _write_json(
        output_dir / "resource.json",
        {
            "schema": "rift_gotcha_step2_p1_raw_bp_resource_v1",
            "point_count": support.point_count,
            "point_chunk_size": point_chunk_size,
            "point_chunks_per_observation": int((support.point_count + point_chunk_size - 1) // point_chunk_size),
            "selected_pulse_count": len(observations),
            "selected_frequency_counts": frequency_counts,
            "kernel_evaluations": preflight["resource_estimate"]["kernel_evaluations"],
            "max_kernel_evaluations": preflight["resource_estimate"]["max_kernel_evaluations"],
            "guard_passed_before_bp": True,
            "elapsed_seconds": elapsed,
            "process_rss_bytes": _process_rss_bytes(),
        },
    )
    _write_json(
        output_dir / "bp_report.json",
        {
            "schema": "rift_gotcha_step2_p1_raw_bp_report_v1",
            "status": "PASS",
            "diagnostic_statement": "conditional raw-BP diagnostic—not calibrated, registered, height, localization, or geometry evidence",
            "title": title,
            "pass": 1,
            "polarization": "HH",
            "sector_id": 2,
            "selected_pulse_count": len(observations),
            "selected_frequency_counts": frequency_counts,
            "native_per_pulse_r0": True,
            "autofocus": "raw/unapplied",
            "phase_hypothesis": CTL.PHASE_HYPOTHESIS_NAME,
            "normalization": "mean_native_sample",
            "z_native_m": 0.0,
            "registration_status": "unresolved",
            "backprojection_metadata": bp.metadata,
            "disclosure": disclosure,
            "plot": plot_metadata,
            "future_path": protocol["future_path"],
        },
    )
    status = {
        "schema": "rift_gotcha_step2_p1_raw_bp_status_v1",
        "status": "PASS",
        "stage": "profile",
        "output_dir": str(output_dir),
        "files": [
            "protocol_echo.json",
            "preflight.json",
            "resource.json",
            "bp_report.json",
            "status.json",
            "bp_complex.npy",
            "x.npy",
            "y.npy",
            "bp_native_xy_z0.png",
        ],
        "test_payload_opened": False,
        "validation_used_for_bp": False,
        "future_auto_chain": False,
    }
    _write_json(output_dir / "status.json", status)
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--archive-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage", required=True)
    args = parser.parse_args(argv)
    run_profile(args.protocol, args.archive_root, args.output_dir, args.stage)
    print(json.dumps({"status": "PASS", "output_dir": str(Path(args.output_dir))}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
