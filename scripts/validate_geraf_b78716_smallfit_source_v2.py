#!/usr/bin/env python3
"""Torch-free preflight for the GeRaF v2 deapodization-cache repair cell.

This validates only authored-source structure.  The actual CUDA regression,
the fresh native-MF target preparation, the 32-update fit, and postflight all
run together in the proposed single PACE allocation.
"""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_B787_ARCHIVE = (
    "/storage/home/hcoda1/1/dbao31/r-jromberg3-0/RIFT/data/"
    "b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz"
)
V1_CELL_ROOT = "/storage/scratch1/1/dbao31/rift_b7873200_geraf_subset16x4_smallfit_v1"
V3_CELL_ROOT = "/storage/scratch1/1/dbao31/rift_b7873200_geraf_subset16x4_smallfit_v3_geometrydepsfix"


class Gates:
    def __init__(self) -> None:
        self.count = 0

    def check(self, condition: bool, detail: str) -> None:
        if not condition:
            raise AssertionError(detail)
        self.count += 1
        print(f"PASS {self.count:02d}: {detail}", flush=True)


def _text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _tree(relative: str) -> ast.AST:
    path = ROOT / relative
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _has_function(tree: ast.AST, name: str) -> bool:
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
        for node in ast.walk(tree)
    )


def _module_path(module: str) -> Path | None:
    relative = Path(*module.split("."))
    module_file = ROOT / relative.with_suffix(".py")
    if module_file.is_file():
        return module_file
    package_file = ROOT / relative / "__init__.py"
    if package_file.is_file():
        return package_file
    package_dir = ROOT / relative
    return package_dir if package_dir.is_dir() else None


def _module_name_for_path(path: Path) -> str:
    relative = path.resolve().relative_to(ROOT.resolve())
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _package_exports(package: Path) -> set[str]:
    """Return names intentionally exported by a package initializer."""

    package_dir = package if package.is_dir() else package.parent
    initializer = package_dir / "__init__.py"
    if not initializer.is_file():
        return set()
    tree = ast.parse(initializer.read_text(encoding="utf-8"), filename=str(initializer))
    exports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            exports.update(alias.asname or alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            exports.update(alias.asname or alias.name for alias in node.names if alias.name != "*")
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
            for target in targets:
                if isinstance(target, ast.Name):
                    exports.add(target.id)
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                exports.add(node.name)
    return exports


def _relative_base(source_path: Path, level: int) -> tuple[str, ...]:
    source_module = _module_name_for_path(source_path)
    module_parts = source_module.split(".") if source_module else []
    package_parts = module_parts if source_path.name == "__init__.py" else module_parts[:-1]
    trim = level - 1
    if trim > len(package_parts):
        return ()
    return tuple(package_parts[: len(package_parts) - trim])


def _resolve_local_import_paths(source_path: Path, tree: ast.AST) -> list[tuple[str, Path]]:
    """Resolve every repository-local import in one file, including relatives."""

    resolved: list[tuple[str, Path]] = []

    def require(module: str, detail: str) -> Path:
        path = _module_path(module)
        if path is None or not path.exists():
            raise AssertionError(f"missing repository-local import {module!r} {detail}")
        resolved.append((module, path.resolve()))
        return path

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name
                if module.startswith(("rift.", "scripts.")) or module in {"rift", "scripts"} or _module_path(module) is not None:
                    require(module, f"from {source_path}")
            continue

        if not isinstance(node, ast.ImportFrom):
            continue

        if node.level:
            base = _relative_base(source_path, node.level)
            if not base:
                raise AssertionError(f"relative import escapes the repository package from {source_path}")
            if node.module:
                module = ".".join((*base, *node.module.split(".")))
                require(module, f"from {source_path}")
            else:
                package = ".".join(base)
                package_path = require(package, f"from {source_path}")
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    candidate = f"{package}.{alias.name}"
                    if _module_path(candidate) is not None:
                        require(candidate, f"from {source_path}")
                    elif alias.name not in _package_exports(package_path):
                        raise AssertionError(
                            f"missing repository-local import {candidate!r} from {source_path}"
                        )
            continue

        module = node.module or ""
        is_known_local_namespace = module.startswith(("rift.", "scripts.")) or module in {"rift", "scripts"}
        if module and (is_known_local_namespace or _module_path(module) is not None):
            module_path = require(module, f"from {source_path}")
            if module in {"rift", "scripts"}:
                exports = _package_exports(module_path)
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    candidate = f"{module}.{alias.name}"
                    if _module_path(candidate) is not None:
                        require(candidate, f"from {source_path}")
                    elif alias.name not in exports:
                        raise AssertionError(
                            f"missing repository-local import {candidate!r} from {source_path}"
                        )

    return resolved


def _check_import_entries(entries: list[tuple[Path, ast.AST]]) -> None:
    pending = [(path.resolve(), tree) for path, tree in entries]
    visited: set[Path] = set()
    while pending:
        path, supplied_tree = pending.pop()
        if path in visited:
            continue
        if path.is_dir():
            initializer = path / "__init__.py"
            visited.add(path)
            if initializer.is_file():
                pending.append((initializer.resolve(), None))
            continue
        if not path.is_file() and supplied_tree is None:
            raise AssertionError(f"missing repository-local source {path}")
        tree = supplied_tree if supplied_tree is not None else ast.parse(
            path.read_text(encoding="utf-8"), filename=str(path)
        )
        visited.add(path)
        for _, imported_path in _resolve_local_import_paths(path, tree):
            if imported_path not in visited:
                pending.append((imported_path, None))


def check_local_import_closure(relative_sources: tuple[str, ...]) -> None:
    """Reject missing repository-local modules across the reachable import graph."""

    _check_import_entries([(ROOT / relative, None) for relative in relative_sources])


def check_local_import_negative_fixtures() -> None:
    """Keep both transitive and package-submodule misses covered without I/O."""

    fixtures = (
        (
            ROOT / "rift" / "_missing_transitive_fixture.py",
            "from .definitely_missing_transitive import value\n",
        ),
        (
            ROOT / "rift" / "_missing_submodule_fixture.py",
            "from rift import definitely_missing_submodule\n",
        ),
    )
    for path, source in fixtures:
        try:
            _check_import_entries([(path, ast.parse(source, filename=str(path)))])
        except AssertionError:
            continue
        raise AssertionError(f"negative import fixture unexpectedly resolved: {path.name}")


def main() -> None:
    gates = Gates()
    range_source = _text("rift/range_operator.py")
    range_tree = _tree("rift/range_operator.py")
    validator_source = _text("scripts/validate_geraf_b78716_smallfit_source_v2.py")
    coherent_source = _text("rift/coherent_radar_geometry.py")
    coherent_tree = _tree("rift/coherent_radar_geometry.py")
    transition = _text("scripts/validate_range_operator_inference_cache_transition.py")
    transition_tree = _tree("scripts/validate_range_operator_inference_cache_transition.py")
    signal_probe = _text("scripts/validate_geraf_b78716_smallfit_signal_path.py")
    signal_probe_tree = _tree("scripts/validate_geraf_b78716_smallfit_signal_path.py")
    timed_wrapper = _text("scripts/run_geraf_b78716_smallfit_timed.py")
    timed_wrapper_tree = _tree("scripts/run_geraf_b78716_smallfit_timed.py")
    driver = _text("train_geraf_smoke.py")
    driver_tree = _tree("train_geraf_smoke.py")
    launcher = _text("slurm/validate_geraf_b78716_smallfit_v2_deapodcachefix.sbatch")

    check_local_import_closure(
        (
            "rift/serialized_range_operator.py",
            "rift/coherent_radar_geometry.py",
            "rift/geraf_b78716_smallfit.py",
            "rift/geraf_signal_operator.py",
            "rift/geraf_b7873200_acquisition.py",
            "rift/geraf_b7873200_protocol.py",
            "rift/geraf_b7873200_source.py",
            "train_geraf.py",
            "train_geraf_smoke.py",
            "scripts/prepare_geraf_b7873200_targets.py",
            "scripts/validate_geraf_b78716_smallfit.py",
            "scripts/validate_range_operator_inference_cache_transition.py",
        )
    )
    check_local_import_negative_fixtures()
    gates.check(
        "def _check_import_entries" in validator_source
        and "def _relative_base" in validator_source
        and "check_local_import_negative_fixtures" in validator_source,
        "recursive local import closure resolves relative imports and catches transitive/package-submodule misses",
    )
    gates.check(
        "BISTATIC_NEAR_FIELD_ABSOLUTE" in coherent_source
        and "MONOSTATIC_NEAR_FIELD_REFERENCE" in coherent_source
        and "MONOSTATIC_FAR_FIELD_REFERENCE" in coherent_source
        and _has_function(coherent_tree, "point_pair_path_lengths")
        and _has_function(coherent_tree, "one_way_range_coordinates")
        and _has_function(coherent_tree, "validate_monostatic_geometry"),
        "the complete shared coherent-radar geometry API is present before Torch fixtures",
    )

    gates.check(
        _has_function(range_tree, "_ordinary_cached_constant")
        and "torch.inference_mode(False)" in range_source
        and "torch.no_grad()" in range_source,
        "shared deapodization constants are constructed or promoted in an ordinary no-grad scope",
    )
    gates.check(
        "regular = _ordinary_cached_constant(cached)" in range_source
        and "if regular is not cached:" in range_source
        and "_DEAPOD_CACHE[key] = regular" in range_source,
        "only an encountered legacy inference cache entry is replaced under its existing key",
    )
    gates.check(
        "cached = _DEAPOD_CACHE.get(key)" in range_source
        and "_DEAPOD_CACHE[key] = d" in range_source
        and "_DEAPOD_CACHE.clear" not in range_source,
        "the established cache key and warm-cache behavior remain in place without a cache reset",
    )
    gates.check(
        _has_function(transition_tree, "main")
        and "nf_full = 600" in transition
        and "oversample = 2" in transition
        and "kernel_width = 20" in transition
        and "matched_filter_from_response_range" in transition
        and "trace_and_match_magnitude" in transition,
        "the allocated CUDA regression uses the failed B787 cache shape and actual preparation-to-GeRaF route",
    )
    gates.check(
        "with torch.inference_mode():" in transition
        and "cold_deapod = range_operator._DEAPOD_CACHE.get(cache_key)" in transition
        and "legacy_inference_deapod" in transition
        and "range_forward_operator_chunks" in transition
        and "range_adjoint_operator_chunks" in transition
        and "BISTATIC_NEAR_FIELD_ABSOLUTE" in transition
        and "serialized_loss" in transition
        and "serialized_re = torch.tensor(" in transition
        and "ordinary_reference = ordinary.detach()" in transition
        and "ordinary_adjoint_reference = ordinary_adjoint.detach()" in transition
        and "serialized_adjoint_positions = rift_positions.detach().clone().requires_grad_(True)" in transition
        and "serialized_weights = torch.complex(serialized_re, serialized_im)" in transition
        and "adjoint_positions" in transition
        and "serialized_adjoint_loss" in transition
        and "import inspect" in transition
        and "def check_chunk_helper_arity" in transition
        and "GERAF_B78716_SMALLFIT_CHUNK_ARITY_PASS" in transition
        and "_DEAPOD_CACHE.clear" not in transition,
        "the regression cheaply binds legacy/extended helper arities, isolates ordinary/serialized forward graphs, then inspects cold inference seeding, legacy promotion, and shared-cache consumers",
    )
    lifecycle_markers = (
        "GERAF_B78716_SMALLFIT_V2_SIGNAL_RESET_PREFLIGHT_START",
        "GERAF_B78716_SMALLFIT_V2_SIGNAL_PATH_START",
        "GERAF_B78716_SMALLFIT_SOURCE_PREFLIGHT_START",
        "GERAF_B78716_SMALLFIT_V2_SOURCE_PREFLIGHT_START",
        "GERAF_B78716_SMALLFIT_TORCH_FIXTURE_START",
        "GERAF_B78716_SMALLFIT_V2_CACHE_TRANSITION_START",
        "GERAF_B78716_SMALLFIT_V2_REAL_COMBINED_START",
        "GERAF_B78716_SMALLFIT_V2_POSTFLIGHT_START",
    )
    marker_offsets = [launcher.index(marker) for marker in lifecycle_markers]
    gates.check(
        marker_offsets == sorted(marker_offsets),
        "the launcher orders source checks, fixtures, exact cache regression, real prepare/fit, and postflight in one allocation",
    )
    gates.check(
        "#SBATCH --signal=TERM@300" in launcher
        and launcher.count("--ignore-signals=TERM") == 2
        and "scripts/run_geraf_b78716_smallfit_timed.py --time-log \"$time_log\" --ready-file \"$driver_ready_file\" -- python" in launcher
        and "scripts/run_geraf_b78716_smallfit_timed.py --time-log \"$signal_time_log\" --ready-file \"$signal_ready_file\" -- python" in launcher
        and "signal_wrapper=" not in launcher
        and "export SLURM_EXPORT_ENV=ALL" in launcher
        and "trap terminal_signal TERM INT" in launcher
        and "run_unresumable_stage()" in launcher
        and "cache_transition_status == 143" in launcher
        and launcher.count("tee_status == 143") == 5
        and "read_gpu_total_mib()" in launcher
        and launcher.count("exit 96") >= 7,
        "the scheduled warning targets a direct timed-wrapper task, while the local srun clients ignore TERM and non-driver signal paths map to terminal evidence",
    )
    gates.check(
        _has_function(timed_wrapper_tree, "main")
        and "start_new_session=True" in timed_wrapper
        and "os.killpg(child.pid, signal.SIGTERM)" in timed_wrapper
        and 'trap "" TERM; exec "$@"' in timed_wrapper
        and "/usr/bin/env" in timed_wrapper
        and "--default-signal=TERM" in timed_wrapper
        and "pending_term" in timed_wrapper
        and timed_wrapper.count("child_is_ready()") >= 3
        and "if pending_term:\n        abort_before_ready" not in timed_wrapper
        and "GERAF_TIMED_WRAPPER_CHILD_READY" in timed_wrapper,
        "the direct wrapper isolates GNU time, restores Python TERM semantics, retains unready stop intent as terminal evidence, and forwards only after readiness",
    )
    gates.check(
        _has_function(signal_probe_tree, "_on_term")
        and _has_function(signal_probe_tree, "_publish_ready")
        and "GERAF_TIMED_WRAPPER_PID" in signal_probe
        and "os.kill(wrapper_pid, signal.SIGTERM)" in signal_probe
        and "signal.getsignal(signal.SIGTERM) != signal.SIG_DFL" in signal_probe
        and "GERAF_TIMED_WRAPPER_CHILD_READY" in signal_probe
        and "GERAF_B78716_SMALLFIT_SIGNAL_PATH_PASS" in signal_probe,
        "the signal-path probe publishes readiness after installing its handler, then exercises direct-wrapper-to-child TERM forwarding without opening data or Torch",
    )
    gates.check(
        _has_function(driver_tree, "_publish_timed_wrapper_ready")
        and "GERAF_TIMED_WRAPPER_READY_FILE" in driver
        and "GERAF_TIMED_WRAPPER_CHILD_READY" in driver
        and "signal.signal(signal.SIGTERM, _request_stop)\n    signal.signal(signal.SIGINT, _request_stop)\n    _publish_timed_wrapper_ready()" in driver,
        "the frozen v1 science driver gains only an opt-in v2 ready marker after its existing TERM and INT handlers are installed",
    )
    gates.check(
        "scripts/validate_geraf_b78716_smallfit_signal_path.py" in launcher
        and "GERAF_B78716_SMALLFIT_V2_SIGNAL_PATH_START" in launcher
        and "GERAF_B78716_SMALLFIT_V2_SIGNAL_PATH_PASS" in launcher
        and "geraf_signal_path_time_v3_geometrydepsfix.txt" in launcher
        and "geraf_signal_path_v3_geometrydepsfix.log" in launcher
        and "signal_rss_kib" in launcher
        and "Maximum resident set size" in launcher
        and "GERAF_TIMED_WRAPPER_FORWARD_TERM child_pgid=" in launcher
        and "GERAF_TIMED_WRAPPER_CHILD_READY" in launcher,
        "the same allocated direct-wrapper fixture proves target readiness, forwarded TERM, a child handler, and a positive GNU-time RSS record before fitting starts",
    )
    gates.check(
        "GERAF_B78716_SMALLFIT_V2_SIGNAL_RESET_PREFLIGHT_START" in launcher
        and "/usr/bin/env --default-signal=TERM -- /usr/bin/true" in launcher
        and "/usr/bin/env" in launcher,
        "the GNU signal-reset capability is checked before the shared timed signal-path fixture and driver run",
    )
    gates.check(
        "resume|postflight" in launcher
        and "GERAF_B78716_SMALLFIT_V2_POSTFLIGHT_RECOVERY_START" in launcher
        and "GERAF_B78716_SMALLFIT_V2_POSTFLIGHT_RECOVERY_PASS" in launcher
        and "completed_driver_logs" in launcher
        and launcher.index("GERAF_B78716_SMALLFIT_V2_POSTFLIGHT_RECOVERY_START")
        < launcher.index("GERAF_B78716_SMALLFIT_SOURCE_PREFLIGHT_START"),
        "a report-plus-final interruption has a narrow postflight-only recovery before any source fixture, target preparation, or update",
    )
    gates.check(
        CANONICAL_B787_ARCHIVE in launcher
        and "NPZ=" in launcher
        and V3_CELL_ROOT in launcher
        and "#SBATCH --job-name=mgr_geraf_b78716_smallfit_v3_geometrydepsfix" in launcher
        and "V3_EXECUTION_ID=geraf_b78716_smallfit_v3_geometrydepsfix" in launcher
        and "GERAF_B78716_SMALLFIT_V3_GEOMETRYDEPSFIX_EXECUTION_ID=$V3_EXECUTION_ID" in launcher
        and "geraf_driver_v3_geometrydepsfix.log" in launcher
        and "geraf_time_v3_geometrydepsfix.txt" in launcher
        and "smallfit_v2_deapodcachefix" not in launcher
        and "mgr_geraf_b78716_smallfit_v2_deapodcachefix" not in launcher
        and "geraf_driver.log" not in launcher
        and "execution_context.log" in launcher,
        "the v3 launcher uses the user-confirmed canonical B787 archive and records a fresh execution identity durably",
    )
    gates.check(
        V1_CELL_ROOT not in launcher
        and "prepared_targets" in launcher
        and "! -e \"$CACHE_ROOT\"" in launcher
        and "range_cache_transition_v3_geometrydepsfix.log" in launcher
        and "GERAF_B78716_SMALLFIT_V2_CACHE_TRANSITION_PASS" in launcher,
        "v3 requires an unused target cache, regenerates its own targets, and retains cache-regression evidence without touching v1 artifacts",
    )
    gates.check(
        "scripts/validate_geraf_b78716_smallfit_source.py" in launcher
        and "scripts/validate_geraf_b78716_smallfit.py" in launcher
        and "scripts/validate_geraf_b78716_smallfit_signal_path.py" in launcher
        and "scripts/run_geraf_b78716_smallfit_timed.py" in launcher
        and "scripts/validate_range_operator_inference_cache_transition.py" in launcher
        and "train_geraf_smoke.py" in launcher
        and "scripts/postflight_geraf_b78716_smallfit.py" in launcher,
        "one proposed allocation contains every required source, fixture, cache-regression, driver, and postflight entrypoint",
    )
    print(f"GERAF_B78716_SMALLFIT_SOURCE_V2_PASS gates={gates.count}", flush=True)


if __name__ == "__main__":
    main()
