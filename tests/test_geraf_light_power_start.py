"""GeRaF configured light_power start (GOTCHA): closed-form TRAIN warm start, legacy recipes unchanged."""
import json

import numpy as np
import pytest
import torch

from rift import geraf_source as source
from rift import geraf_source_training as runtime
from tests.test_geraf_source import SMALL, TinyData

WARM = dict(SMALL, light_power_start='train_warm_start_16_views')


def test_recipe_declares_the_start_and_legacy_recipes_omit_it():
    assert 'light_power_start' not in source.recipe_from_config(SMALL, .15)
    assert source.recipe_from_config(WARM, .15)['light_power_start'] == 'train_warm_start_16_views'
    with pytest.raises(ValueError):
        source.recipe_from_config(dict(SMALL, light_power_start='fit'), .15)


def test_warm_start_matches_the_initial_render_to_the_train_targets_without_touching_the_rng(tmp_path):
    data = TinyData()
    recipe = source.recipe_from_config(WARM, data.extent)
    targets = runtime.SourceTargets(tmp_path / 'targets', data, recipe)
    targets.start()
    accumulated = {h: targets.accumulated(h, 'cpu', lambda: False) for h in data.heads}
    np.random.seed(7)
    torch.manual_seed(0)
    models = {h: source.build_model(recipe) for h in data.heads}
    for model in models.values():
        model.update_step(source.source_step(recipe, 0))
    state = np.random.get_state()
    first = runtime.light_power_warm_start(models, data, recipe, targets, accumulated, 'cpu', lambda: False)
    after = np.random.get_state()
    assert all(np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b for a, b in zip(state, after))
    assert first['scalar']['light_power'] == pytest.approx(float(np.log(first['scalar']['scale'])))
    again = runtime.light_power_warm_start(models, data, recipe, targets, accumulated, 'cpu', lambda: False)
    assert again['scalar']['scale'] == pytest.approx(1.0, rel=1e-6)      # already at the least-squares scale


def test_training_records_the_warm_start_and_legacy_training_does_not(tmp_path):
    for config, name in ((WARM, 'warm'), (SMALL, 'legacy')):
        assert runtime.train(data=TinyData(), output_dir=tmp_path / name, config=config, device='cpu')['status'] == 'complete'
    record = json.loads((tmp_path / 'warm/light_power_warm_start.json').read_text())
    assert record['scalar']['light_power'] != 0.0
    assert not (tmp_path / 'legacy/light_power_warm_start.json').exists()
