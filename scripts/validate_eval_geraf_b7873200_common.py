#!/usr/bin/env python3
"""Static and tensor-level checks for the GeRaF common evaluator.

The static checks run with the repository's lightweight local runtime.  The
tensor checks run when the project PyTorch environment is available; they are
also suitable for the PACE preflight environment.
"""

from __future__ import annotations

import ast
import math
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "eval_geraf_complex_response.py"


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(name + (f": {detail}" if detail else ""))
    print(f"PASS: {name}" + (f" -- {detail}" if detail else ""))


def expect_error(name: str, callback, fragment: str) -> None:
    try:
        callback()
    except (TypeError, ValueError, AssertionError) as exc:
        check(name, fragment in str(exc), str(exc))
    else:
        raise AssertionError(f"{name}: expected an exception containing {fragment!r}")


def static_checks() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    launcher = (ROOT / "slurm" / "eval_geraf_b7873200_common_v1.sbatch").read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(SOURCE))
    check("evaluator parses", tree is not None)
    check("no global inference mode", "@torch.inference_mode()" not in source)
    check("normal render enables gradients", "with torch.enable_grad():" in source)
    check("response-only renderer is used", "def render_complex_response(" in source)
    check("response-only renderer uses range operator", "pairwise_range_forward_operator(" in source)
    check("unused matched-filter adjoint is absent", "predict_normalized_magnitude_with_response(" not in source)
    check("checkpoint history cadence is validated", "expected_steps = list(range(1_000, SELECTED_CHECKPOINT_STEP + 1, 1_000))" in source)
    check("checkpoint acquisition provenance is compared", "acquisition_records_equal(checkpoint.get(\"acquisition_record\", {}), cache.acquisition_record)" in source)
    check("checkpoint run identity is recomputed", "tg.run_identity(args=args, cache=cache, model_config=checkpoint[\"model_config\"])" in source)
    check("response flattening uses Tx/Rx permutation", ".permute(2, 1, 0)" in source)
    check("raw response preserves full chirp axis", "raw_measured = response_source.response_view(view_index)" in source
          and "raw_measured.mean(axis=2)" in source)
    check("common normalized power helper reused", "normalize_power_db" in source)
    check("common range-power target helper reused", "response_view_to_range_power" in source)
    check("ROI helper reused", "scene_range_mask" in source)
    check("no extra gain contract", '"calibration": "no extra gain"' in source)
    check("reserved roles are explicitly reported", '"reserved_test_accessed": role == "test"' in source)
    check("production sample gate is present", "EXPECTED_VALIDATION_SAMPLES" in source)
    check(
        "resume completion requires both metric fields",
        'np.isfinite(cache["coherent_rel_mse"]) & np.isfinite(cache["range_power_rel_mse"])' in source,
    )
    check("resume identity is persisted", "_cache_identity_path" in source)
    check("launcher uses evaluator", (ROOT / "slurm" / "eval_geraf_b7873200_common_v1.sbatch").is_file())
    check("launcher pins Embers account", "#SBATCH --account=gts-jromberg3-ece" in launcher and 'SLURM_JOB_ACCOUNT:-}" == "gts-jromberg3-ece"' in launcher)
    check("launcher pins gpu-h100", "#SBATCH --partition=gpu-h100" in launcher and 'SLURM_JOB_PARTITION:-}" == "gpu-h100"' in launcher)
    check("launcher runs one-view smoke first", "--max-views 1" in launcher and "SMOKE_ROOT" in launcher)
    check("launcher reuses validated smoke evidence", "SMOKE_IDENTITY" in launcher and "incomplete_or_smoke" in launcher and "GERAF_B7873200_COMMON_EVAL_SMOKE_REUSE_PASS" in launcher)
    check("launcher runs full production after smoke", launcher.index("--max-views 1") < launcher.index("--save-every 5"))
    check("launcher does not request Inferno", "#SBATCH --qos=inferno" not in launcher.lower() and "#SBATCH --partition=inferno" not in launcher.lower())


def tensor_checks() -> bool:
    try:
        import torch
    except ModuleNotFoundError:
        print("SKIP: tensor checks require the project PyTorch runtime")
        return False

    sys.path.insert(0, str(ROOT))
    from scripts.eval_geraf_complex_response import (  # noqa: WPS433
        EXPECTED_POWER_ELEMENT_COUNT,
        EXPECTED_VALIDATION_SAMPLES,
        _aggregate,
        flatten_ge_ra_f_response,
        validate_production_aggregate,
    )

    response = torch.zeros((600, 16, 16), dtype=torch.complex64)
    response[300, 1, 2] = complex(12345.0, 0.0)
    response[599, 0, 0] = complex(9876.0, 0.0)
    flattened = flatten_ge_ra_f_response(response)
    check("flattened shape is pair-by-frequency", tuple(flattened.shape) == (256, 600))
    check("Tx-outer/Rx-inner flattening", flattened[2 * 16 + 1, 300].item() == complex(12345.0, 0.0))
    check("channel flattening preserves first pair", flattened[0, 599].item() == complex(9876.0, 0.0))

    cache = {
        "coherent_rel_mse": torch.tensor([0.25, 0.5]).numpy(),
        "range_power_rel_mse": torch.tensor([0.1, 0.2]).numpy(),
        "coherent_squared_error": torch.tensor([2.0, 3.0]).numpy(),
        "coherent_target_energy": torch.tensor([8.0, 6.0]).numpy(),
        "coherent_prediction_energy": torch.tensor([10.0, 12.0]).numpy(),
        "coherent_sample_count": torch.tensor([4.0, 4.0]).numpy(),
        "range_power_squared_error": torch.tensor([1.0, 3.0]).numpy(),
        "range_power_target_squared_norm": torch.tensor([10.0, 10.0]).numpy(),
        "range_power_element_count": torch.tensor([4.0, 4.0]).numpy(),
        "elapsed_seconds": torch.tensor([1.0, 2.0]).numpy(),
    }
    aggregate = _aggregate(cache)
    check("pooled coherent aggregation", math.isclose(float(aggregate["coherent_complex_rel_mse"]), 5.0 / 14.0))
    check("pooled power aggregation", math.isclose(float(aggregate["normalized_range_power_rel_mse"]), 0.2))

    identity = {"view_indices": [1, 2, 3]}
    check("production constants are frozen", EXPECTED_VALIDATION_SAMPLES == 153_600_000 and EXPECTED_POWER_ELEMENT_COUNT == 3_328_000)
    expect_error(
        "incomplete production aggregate rejected",
        lambda: validate_production_aggregate({"views_complete": 2, "views_total": 3}),
        "all 1000 validation views",
    )
    del identity
    return True


def main() -> None:
    static_checks()
    tensor_checks()
    print("GERAF_B7873200_COMMON_EVAL_VALIDATION_PASS")


if __name__ == "__main__":
    main()
