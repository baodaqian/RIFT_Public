"""PVC twin of tests/test_geraf_far_range.py for rift_pvc.geraf_source."""
from rift_pvc.geraf_source import (LEGACY_RECEIVER_GEOMETRY, NativeGeRaFStage1, SingleBankGeRaFStage1,
                                   build_model, recipe_from_config)
from rift_pvc.vendor.geraf_sens.rf_rendering import GeRaFStage1
from tests.test_geraf_far_range import check_lane


def test_receiver_geometry_pvc_lane(monkeypatch):
    monkeypatch.setenv('RIFT_ACCELERATOR', 'cpu')
    check_lane(recipe_from_config, build_model, NativeGeRaFStage1, SingleBankGeRaFStage1, GeRaFStage1,
               LEGACY_RECEIVER_GEOMETRY)
