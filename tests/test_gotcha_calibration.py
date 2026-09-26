"""GOTCHA calibration-array method: exact on a synthetic point reflector."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from rift.gotcha_calibration import (GOTCHA_HH_SYSTEM_CONSTANT, boresight_azimuth, boresight_sectors,
                                     measure_trihedral, system_constant, triangular_trihedral_rcs)
from rift.gotcha_dataset import C


def pulses(target, amplitude, count=60, radius=10000.0):
    f = np.linspace(9.288e9, 9.910e9, 64)
    rng = np.random.default_rng(0)
    out = []
    for k in range(count):
        az = np.radians(270 + (k - count / 2) * 0.1)
        a = np.array([radius * math.cos(az) * 0.7, radius * math.sin(az) * 0.7, radius * 0.7])
        r0 = float(np.linalg.norm(a)) + rng.normal(0, 0.1)
        d = np.linalg.norm(np.asarray(target) - a) - r0
        response = amplitude * np.exp(-4j * math.pi * f * d / C) * np.exp(1j * 0.3)
        out.append(SimpleNamespace(position_m=a, reference_range_m=r0, frequencies_hz=f, response=response))
    return out


def test_point_reflector_amplitude_offset_and_constant_are_recovered():
    target = (-7.5, 51.0, -0.1)
    obs = pulses(target, 3e-4)
    result = measure_trihedral(obs, (-7.5, 51.3, -0.1))           # tabulated position 0.3 m off
    assert result['amplitude'] == pytest.approx(3e-4, rel=1e-3)
    assert result['offset_m'][:2] == pytest.approx([0.0, -0.3], abs=0.011)
    assert result['background_amplitude'] < 0.2 * result['amplitude']
    rcs = triangular_trihedral_rcs(15 * 0.0254, C / 9.6e9)
    assert 10 * math.log10(rcs) == pytest.approx(19.57, abs=0.01)
    assert system_constant(1.0, 10.0, 4.0) == pytest.approx(400 / 2)


def test_boresight_convention_and_pinned_constant():
    assert boresight_azimuth(180) == 270 and boresight_azimuth(0) == 90 and boresight_azimuth(270) == 180
    assert boresight_sectors(range(1, 361), 180) == list(range(265, 277))
    assert 7000 < GOTCHA_HH_SYSTEM_CONSTANT < 9000
