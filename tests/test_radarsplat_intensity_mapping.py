"""RadarSplat image intensity domain (user decision 3, 2026-09-22): log-power targets, legacy linear resumes."""
import signal

import numpy as np
import pytest
import torch

from rift import radarsplat_release as release
from rift.radar_fields_dataset import normalize_power_db


def test_log_mapping_is_the_radar_fields_mapping_and_linear_is_unchanged():
    power = torch.rand(7, 9, dtype=torch.float64)**6 * 3.0
    peak = float(power.max())
    log = release.intensity(power, peak, release.LOG_INTENSITY)
    torch.testing.assert_close(log, normalize_power_db(power, peak, 60.0), rtol=0, atol=0)
    np.testing.assert_allclose(release.intensity(power.numpy(), peak, release.LOG_INTENSITY), log.numpy(),
                               rtol=0, atol=1e-15)
    torch.testing.assert_close(release.intensity(power, peak), power/peak, rtol=0, atol=0)
    assert float(log.min()) >= 0 and float(log.max()) == 1.0
    with pytest.raises(ValueError):
        release.intensity(power, peak, "per_view_peak")


def test_linear_identity_is_the_earlier_one_and_the_cache_recipe_is_shared():
    kwargs = dict(dataset_identity="d", target_recipe={"schema": "cache"}, train_peak=2.0, half_extent_m=5.0,
                  profile="budget48")
    linear = release.recipe(**kwargs)
    log = release.recipe(**kwargs, intensity_mapping=release.LOG_INTENSITY)
    assert "intensity_mapping" not in linear
    assert linear["adapter"]["observable"] == "clipped train-peak-normalized native MF power"
    assert release.intensity_mapping_from_identity(linear) == release.LINEAR_INTENSITY
    assert log["intensity_mapping"] == release.LOG_INTENSITY == release.DEFAULT_INTENSITY
    # Only the image domain differs: the target cache and every model constant are shared.
    for key in set(linear) - {"adapter"}:
        assert log[key] == linear[key]
    assert {k: v for k, v in log["adapter"].items() if k != "observable"} == \
           {k: v for k, v in linear["adapter"].items() if k != "observable"}


def _synthetic(tmp_path, monkeypatch):
    from scripts.validate_radarsplat_b7873200_native_contract import _make_synthetic_cache
    from rift.radarsplat_b7873200_protocol import load_cache
    original_to = torch.Tensor.to
    monkeypatch.setattr(torch.Tensor, "to", lambda t, *a, **kw:
                        t if a and isinstance(a[0], str) and a[0] == "cuda" else original_to(t, *a, **kw))
    _make_synthetic_cache(tmp_path/"cache")
    return load_cache(tmp_path/"cache")


def test_a_quiet_view_keeps_occupancy_labels_only_in_the_log_domain(tmp_path, monkeypatch):
    """The audit's B787 case: a view whose brightest pixel is 0.0072 of the TRAIN peak."""
    import rift.radarsplat_b7873200_protocol as protocol
    from rift.radarsplat_fidelity import OccupancyRecipe, TrainingOccupancy
    cache = _synthetic(tmp_path, monkeypatch)
    load = protocol.load_target
    def quiet(*args, **kwargs):
        arrays = dict(load(*args, **kwargs))
        power = arrays["radarsplat_mf_power"].astype(np.float64)
        arrays["radarsplat_mf_power"] = power/power.max()*0.0072*cache.train_peak_power
        return arrays
    monkeypatch.setattr(protocol, "load_target", quiet)
    index = cache.train_indices[0]
    peaks = {}
    for mapping in release.INTENSITY_MAPPINGS:
        occupancy = TrainingOccupancy(cache, OccupancyRecipe(window_views=11, power_threshold=.10),
                                      intensity_mapping=mapping)
        _, power, _ = occupancy._donor(index)
        peaks[mapping] = float(np.max(power))
    assert peaks[release.LINEAR_INTENSITY] < .10 < peaks[release.LOG_INTENSITY]
    assert peaks[release.LOG_INTENSITY] == pytest.approx(1 + 10*np.log10(0.0072)/60, abs=0.05)


def test_fresh_runs_take_log_and_saved_runs_resume_in_their_own_domain(tmp_path, monkeypatch):
    from rift import radarsplat_release_training as engine
    from rift.radarsplat_fidelity import release_ssim_index
    import train_radarsplat as lifecycle
    cache = _synthetic(tmp_path, monkeypatch)
    original_create = release.create_scene
    monkeypatch.setattr(engine, "create_scene", lambda **kw: original_create(**{**kw, "num_points": 2}))
    control = dict(calls=0)
    class Renderer:
        def __init__(self, source, units): self.units = units
        def __call__(self, splats, pose, grid, degree, bg):
            control["calls"] += 1
            if control["calls"] == 1:
                signal.raise_signal(signal.SIGTERM)
            size = (grid.output_azimuth_bins, grid.num_range_bins)
            return splats["sh0"].mean().sigmoid().expand(size), splats["opacities"].sigmoid().mean().expand(size)
    monkeypatch.setattr(engine, "ReleasedRenderer", Renderer)
    ssim = lambda x, y, padding: release_ssim_index(x[0, 0], y[0, 0])
    def step(folder, **kwargs):
        control["calls"] = 0
        with pytest.raises(SystemExit):
            engine.train(cache, folder, device=torch.device("cpu"), rendering=object(), fused_ssim=ssim, **kwargs)
        return lifecycle._load_checkpoint(folder/"checkpoint_latest.pt", torch.device("cpu"))
    fresh = step(tmp_path/"fresh")
    assert fresh["identity"]["intensity_mapping"] == release.LOG_INTENSITY
    legacy = step(tmp_path/"legacy", intensity_mapping=release.LINEAR_INTENSITY)
    assert "intensity_mapping" not in legacy["identity"]
    assert legacy["identity"] == engine.identity_for_cache(cache, "upstream")
    resumed = step(tmp_path/"legacy")        # no mapping given: the saved (linear) one continues
    assert resumed["step"] == legacy["step"] + 1 and resumed["identity"] == legacy["identity"]
    with pytest.raises(ValueError, match="identity mismatch"):
        engine.train(cache, tmp_path/"legacy", device=torch.device("cpu"), rendering=object(), fused_ssim=ssim,
                     intensity_mapping=release.LOG_INTENSITY)
