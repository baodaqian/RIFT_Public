"""GeRaF receiver geometry at GOTCHA range: float64 sphere intersection, legacy float32 cast kept exact."""
import pytest
import torch

from rift.geraf_source import (LEGACY_RECEIVER_GEOMETRY, NativeGeRaFStage1, SingleBankGeRaFStage1,
                               build_model, recipe_from_config)
from rift.vendor.geraf_sens.rf_rendering import GeRaFStage1


def geometry(radius, seed=0, count=10000):
    g = torch.Generator().manual_seed(seed)
    points = torch.randn(count, 3, generator=g)
    points = points / points.norm(dim=-1, keepdim=True) * torch.rand(count, 1, generator=g) ** (1 / 3)
    receiver = torch.tensor([[0.6, 0.0, 0.8]], dtype=torch.float64) * radius
    return receiver, points.reshape(100, 100, 3).float()


def check_lane(recipe_from_config, build_model, native, single, stage1, legacy):
    recipe = recipe_from_config({}, 5.0)
    assert recipe['receiver_geometry'] == 'float64_intersection'
    assert 'receiver_geometry' not in recipe_from_config({'receiver_geometry': legacy}, 5.0)
    with pytest.raises(ValueError):
        recipe_from_config({'receiver_geometry': 'float16'}, 5.0)
    fixed = build_model(recipe)
    old = build_model(recipe_from_config({'receiver_geometry': legacy}, 5.0))
    assert type(fixed) is native and type(build_model(recipe_from_config({'bank_size': 1}, 5.0))) is single
    # GOTCHA Camry (10145 m / 5 m), the collection (10 m / 0.15 m) and a near case.
    for radius, far in ((2029.0, True), (66.7, False), (2.0, False)):
        receiver, points = geometry(radius)
        reference_valid, reference_points = stage1.intersect_sphere(fixed, receiver.double(), points.double())
        valid, hits = fixed.intersect_sphere(receiver, points)
        assert torch.equal(valid, reference_valid) and hits.dtype == torch.float32
        torch.testing.assert_close(hits, reference_points.float(), rtol=0, atol=1e-6)
        old_valid, old_hits = old.intersect_sphere(receiver, points)
        release_valid, release_hits = stage1.intersect_sphere(old, receiver.float(), points)
        assert torch.equal(old_valid, release_valid) and torch.equal(old_hits, release_hits)
        if far:
            assert bool((old_valid != reference_valid).any())      # the release cast's false misses
        else:
            # No misses; the collection's float32 hits stay within a few mm (0.05 of the 0.15 m scene radius).
            assert torch.equal(old_valid, reference_valid)
            assert float((old_hits - hits).abs().max()) < (5e-2 if radius > 10 else 1e-5)


def test_receiver_geometry_cuda_lane():
    check_lane(recipe_from_config, build_model, NativeGeRaFStage1, SingleBankGeRaFStage1, GeRaFStage1,
               LEGACY_RECEIVER_GEOMETRY)
