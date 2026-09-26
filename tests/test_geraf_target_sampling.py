"""Bounded synthetic checks against the original GeRaF interpolation."""
import numpy as np
import pytest

from rift.geraf_target_sampling import sample_lattice
from rift.vendor.geraf_sens.loading import _sample_volume


def compare(volume, points):
    calls = []

    def fetch(ids):
        calls.append(ids.copy())
        assert ids.dtype == np.int64
        np.testing.assert_array_equal(ids, np.unique(ids))
        assert np.all((0 <= ids) & (ids < volume.size))
        return volume.reshape(-1)[ids]

    actual = sample_lattice(points, volume.shape[0], fetch)
    expected = np.asarray(_sample_volume(volume, points, "bilinear", True)).reshape(-1)
    assert actual.shape == (len(points),)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=5e-7)
    assert len(calls) <= 1
    if calls:
        assert len(calls[0]) <= 8 * len(points)
    return actual, calls


@pytest.mark.parametrize("size", [5, 17, 101])
def test_random_queries_match_source(size):
    rng = np.random.default_rng(2026 + size)
    volume = rng.standard_normal((size,) * 3).astype(np.float32)
    points = rng.uniform(-1.15, 1.15, (4096, 3))
    compare(volume, points)


@pytest.mark.parametrize("size", [5, 17, 101])
def test_endpoints_zero_padding_and_partial_outside(size):
    volume = np.arange(size**3, dtype=np.float32).reshape((size,) * 3) / np.float32(size**3)
    edge = 2 / (size - 1)
    points = np.array([[0, 0, 0], [-1, -1, -1], [1, 1, 1],
                       [-1-edge/4, .3, -.4], [1+edge/4, .3, -.4],
                       [-1-edge, 0, 0], [1+edge, 0, 0],
                       [0, -1-edge/3, .6], [0, 0, 1+edge/2],
                       [3, 3, 3], [-3, -3, -3]], dtype=np.float64)
    compare(volume, points)


def test_coordinates_quantize_to_fp32_before_lattice_conversion():
    rng = np.random.default_rng(17)
    volume = rng.standard_normal((101,) * 3).astype(np.float32)
    # Adjacent float64 numbers collapse to the same FP32 source coordinate.
    x = np.float64(np.float32(.183746))
    points = np.array([[x, .19, -.37], [np.nextafter(x, np.inf), .19, -.37]])
    actual, _ = compare(volume, points)
    assert actual[0] == actual[1]


def test_duplicate_queries_fetch_unique_corners_once_in_xyz_order():
    volume = np.arange(5**3, dtype=np.float32).reshape((5,) * 3)
    points = np.repeat([[.125, -.375, .625]], 10, axis=0)
    _, calls = compare(volume, points)
    expected = np.array([(x*5+y)*5+z for x in (2, 3) for y in (1, 2) for z in (3, 4)])
    np.testing.assert_array_equal(calls[0], expected)
    _, calls = compare(volume, np.array([[-1, -1, -1], [1, 1, 1]], dtype=float))
    np.testing.assert_array_equal(calls[0], [0, 124])


@pytest.mark.parametrize("points", [np.empty((0, 3)), np.array([[2., 0., 0.], [0., -4., 0.], [1e30, 0., 0.]])])
def test_empty_or_all_outside_does_not_fetch(points):
    def forbidden(_):
        raise AssertionError("padding must not request target values")
    np.testing.assert_array_equal(sample_lattice(points, 101, forbidden), np.zeros(len(points), np.float32))


def test_storage_and_callback_scale_with_queries_not_lattice_volume():
    calls = []
    def fetch(ids):
        calls.append(ids.copy())
        assert len(ids) <= 24
        return np.ones(len(ids), np.float32)
    # A dense FP32 lattice of this size would require 4 exabytes.
    actual = sample_lattice(np.array([[.12345, -.2371, .9111], [0., 0., 0.], [-1., 1., -1.]]), 1_000_000, fetch)
    np.testing.assert_allclose(actual, 1., atol=1e-7)
    assert len(calls) == 1


@pytest.mark.parametrize("size", [True, 1, 2.5, 3_000_000])
def test_invalid_grid_rejected(size):
    with pytest.raises(ValueError, match="grid_size"):
        sample_lattice(np.zeros((1, 3)), size, lambda ids: np.ones(len(ids), np.float32))


@pytest.mark.parametrize("points", [np.zeros(3), np.zeros((1, 2)), [[np.nan, 0, 0]], [[np.inf, 0, 0]], [[1e300, 0, 0]], [[1j, 0, 0]]])
def test_invalid_query_rejected(points):
    with pytest.raises(ValueError, match="points_norm"):
        sample_lattice(points, 5, lambda ids: np.ones(len(ids), np.float32))


@pytest.mark.parametrize("value", [np.array([np.nan], np.float32), np.array([np.inf], np.float32),
                                    np.ones(1, np.float64), np.ones((1, 1), np.float32)])
def test_invalid_callback_values_rejected(value):
    with pytest.raises(ValueError, match="fetch_values"):
        sample_lattice(np.array([[-1., -1., -1.]]), 5, lambda ids: value)
