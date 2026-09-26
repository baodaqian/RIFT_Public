"""Numerical comparisons must use independent copies of GeRaF's mutable bank."""
import copy

import torch

from scripts_pvc.audit_ab_devices import SMALL, geraf_case, pvc_geraf
from tests.test_geraf_source import compare_nested


def test_repeated_comparison_preserves_initial_bank_and_chunk_cursor(monkeypatch):
    monkeypatch.setenv("RIFT_ACCELERATOR", "cpu")
    torch.manual_seed(42)
    state = copy.deepcopy(pvc_geraf.build_model(
        pvc_geraf.recipe_from_config(SMALL, .15), "cpu").state_dict())
    pristine = copy.deepcopy(state)
    first = geraf_case(pvc_geraf, state, torch.device("cpu"))
    compare_nested(state, pristine)
    second = geraf_case(pvc_geraf, state, torch.device("cpu"))
    compare_nested(state, pristine)
    compare_nested(first, second)
