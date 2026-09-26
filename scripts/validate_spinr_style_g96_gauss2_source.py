#!/usr/bin/env python3
"""Torch-free source and numerical checks for the isolated G96 candidate.

The checks exercise the cell-rule constructor on tiny fixtures only.  They do
not instantiate the neural model, open B787 data, launch a job, or claim
quadrature convergence.
"""

from __future__ import annotations

import ast
import importlib.util
import math
from pathlib import Path
import sys

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

_SPEC = importlib.util.spec_from_file_location(
    "spinr_quadrature_torch_free_g96", PROJECT / "rift" / "spinr_quadrature.py"
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError("could not load the Torch-free quadrature module")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
gauss_legendre_cell_grid_arrays = _MODULE.gauss_legendre_cell_grid_arrays

CHECKS = 0


def check(condition: bool, message: str) -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(message)
    print(f"PASS: {message}")


def parse(relative: str) -> str:
    path = PROJECT / relative
    source = path.read_text(encoding="utf-8")
    ast.parse(source, filename=str(path))
    return source


def source_contract() -> None:
    driver = parse("train_spinr_style_smoke.py")
    base_driver = parse("rift/spinr_smoke_runtime.py")
    quadrature = parse("rift/spinr_quadrature.py")
    launcher = (PROJECT / "slurm" / "validate_spinr_style_b78716_smallfit_g96_gauss2_v1.sbatch").read_text(
        encoding="utf-8"
    )
    check("G96_PARENT_CELL_GRID_SIZE = 96" in driver, "candidate fixes the parent support grid at G=96")
    check("G96_TRAINING_NODES_PER_CELL = SPINR_STYLE_GAUSS2_NODES_PER_CELL" in driver,
          "candidate uses two Gauss--Legendre nodes per axis for training")
    check("G96_REFERENCE_NODES_PER_CELL = SPINR_STYLE_GAUSS3_NODES_PER_CELL" in driver,
          "candidate uses three Gauss--Legendre nodes per axis for the independent reference")
    check("G96_TRAINING_EFFECTIVE_GRID_SIZE" in driver and "G96_REFERENCE_EFFECTIVE_GRID_SIZE" in driver,
          "candidate records separate effective 192^3 and 288^3 grid identities")
    check("G96_TRAINING_RULE" in driver and "G96_REFERENCE_RULE" in driver,
          "candidate records the exact G96/Gauss2-to-Gauss3 rule pair")
    check("G96_NUMERICAL_RECIPE_ID" in driver and "G96_SMALLFIT_RUN_NAME" in driver,
          "candidate has distinct recipe, checkpoint, and run identities")
    check("gauss3_v1" not in driver and "gauss2_v2" not in driver,
          "candidate does not reuse completed Gauss2 or Gauss3 artifact identities")
    check("training_rule_is_proposed_correction" in driver and "reference_is_independent_diagnostic_only" in driver,
          "candidate distinguishes a proposed integration repair from convergence evidence")
    check("gauss_legendre_cell_grid(" in driver and "def gauss_legendre_cell_grid_arrays(" in quadrature,
          "candidate routes through physical cell quadrature with Torch-free coverage")
    check("signal_gate" in driver and "directional_gradient_gate" in driver
          and "quadrature_difference_decision" in driver,
          "candidate retains independent signal and directional-gradient gates")
    check("CANONICAL_VIEW_BATCH" in driver and "SMALLFIT_MAX_UPDATES" in base_driver,
          "candidate preserves the four-view batch and bounded update clock from the v1 driver")
    check("validate_spinr_style_g96_gauss2_source.py" in launcher
          and "validate_spinr_style_g96_gauss2_runtime_preflight.py" in launcher
          and "train_spinr_style_smoke.py" in launcher,
          "one launcher orders source check, runtime fixture, and isolated candidate fit")
    check("resume_requested=false" in launcher
          and "NO_DURABLE_CHECKPOINT_FRESH_RESTART" in launcher
          and 'resume_args=(--resume "$latest_checkpoint")' in launcher,
          "launcher distinguishes normal resume from explicit fresh restart after a pre-checkpoint interruption")
    check("NO_DURABLE_CHECKPOINT_FRESH_RESTART" in launcher
          and launcher.index("NO_DURABLE_CHECKPOINT_FRESH_RESTART") < launcher.index("resume_args=()", launcher.index("NO_DURABLE_CHECKPOINT_FRESH_RESTART")),
          "pre-checkpoint restart clears resume arguments before invoking the driver")
    check('[[ ! -e "$RUN_ROOT" ]] || exit 94' in launcher
          and '[[ ! -e "$final_checkpoint" && ! -e "$report" && ! -e "$postflight" ]] || exit 94' in launcher
          and '[[ ! -e "$launcher_log_root" || -d "$launcher_log_root" ]] || exit 94' in launcher
          and '[[ "$prior_entry" == "$launcher_log_root" ]] || exit 94' in launcher,
          "terminal, mixed-artifact, and non-log residue states remain fail-closed")
    check('attempt_log_dir="$launcher_log_root/${SLURM_JOB_ID}"' in launcher
          and '[[ ! -e "$attempt_log_dir" ]] || exit 94' in launcher,
          "each attempt still requires a unique current launcher-log directory")
    check("#SBATCH --qos=inferno" in launcher and "#SBATCH --time=12:00:00" in launcher
          and "#SBATCH --mem=32G" in launcher and "#SBATCH --cpus-per-task=6" in launcher,
          "launcher retains the bounded six-CPU/32-GiB Inferno 12-hour envelope")


def numerical_rule_regression() -> None:
    support = 0.15
    parent_grid = 2
    domain_volume = (2.0 * support) ** 3
    points2, weights2 = gauss_legendre_cell_grid_arrays(
        parent_grid, nodes_per_cell=2, support_m=support
    )
    points3, weights3 = gauss_legendre_cell_grid_arrays(
        parent_grid, nodes_per_cell=3, support_m=support
    )
    check(points2.shape == (4**3, 3) and weights2.shape == (4**3,),
          "two-node training rule has the expected tensor shape on a tiny parent grid")
    check(points3.shape == (6**3, 3) and weights3.shape == (6**3,),
          "three-node reference rule has the expected tensor shape on a tiny parent grid")
    check(np.isfinite(points2).all() and np.isfinite(weights2).all() and np.all(weights2 > 0.0),
          "training points and physical weights are finite and strictly positive")
    check(np.isfinite(points3).all() and np.isfinite(weights3).all() and np.all(weights3 > 0.0),
          "reference points and physical weights are finite and strictly positive")
    check(math.isclose(float(weights2.sum()), domain_volume, rel_tol=0.0, abs_tol=1e-15),
          "two-node physical weights integrate a constant over the support")
    check(math.isclose(float(weights3.sum()), domain_volume, rel_tol=0.0, abs_tol=1e-15),
          "three-node physical weights integrate a constant over the support")
    polynomial = points3[:, 0] ** 4 * points3[:, 1] ** 2 * points3[:, 2] ** 2
    expected = (2.0 * support**5 / 5.0) * (2.0 * support**3 / 3.0) ** 2
    check(math.isclose(float(np.dot(weights3, polynomial)), expected, rel_tol=0.0, abs_tol=1e-13),
          "three-node reference exactly integrates the degree-four fixture")


def main() -> int:
    source_contract()
    numerical_rule_regression()
    print(f"SpINR-style G96 source/numeric validation passed: {CHECKS} checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
