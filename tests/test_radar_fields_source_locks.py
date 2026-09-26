"""source-adapted-v3 accepts only the released RadarField model and its declared adaptation constants (RF2)."""
from types import SimpleNamespace

import pytest

from rift.radar_fields_gotcha import DEFAULTS, recipe_from_config
from rift.radar_fields_recipe import AUDITED_RECIPE, SOURCE_LOCKS, SOURCE_RECIPE, recipe_contract


def changed(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, str):
        return 'torch'
    return value * 2 if value else 1


def source_args(**override):
    return SimpleNamespace(**{**SOURCE_LOCKS, **override}, recipe=SOURCE_RECIPE, ray_samples=10)


def test_release_values_pass_and_every_locked_override_is_rejected():
    assert recipe_contract(source_args())['recipe'] == SOURCE_RECIPE
    for key, value in SOURCE_LOCKS.items():
        with pytest.raises(ValueError, match=f'source-adapted-v3 fixes {key}='):
            recipe_contract(source_args(**{key: changed(value)}))


def test_gotcha_recipe_rejects_every_locked_config_override():
    recipe = recipe_from_config({}, 5.0, 1500)
    assert recipe['controls']['profile'] == SOURCE_RECIPE
    for key in sorted(set(DEFAULTS) & set(SOURCE_LOCKS)):
        with pytest.raises(ValueError, match=f'source-adapted-v3 fixes {key}='):
            recipe_from_config({key: changed(SOURCE_LOCKS[key])}, 5.0, 1500)
    recipe_from_config({'profile': AUDITED_RECIPE, 'hidden_dim': 128}, 5.0, 1500)


def test_pvc_torchshim_contract_passes_release_values_and_rejects_overrides(monkeypatch):
    monkeypatch.setenv('RIFT_ACCELERATOR', 'cpu')
    from rift_pvc import radar_fields_training as twins
    args = source_args(model_backend=twins.TORCHSHIM_BACKEND)
    assert twins.recipe_contract(args)['model_backend'] == twins.TORCHSHIM_BACKEND
    with pytest.raises(ValueError, match='source-adapted-v3 fixes hidden_dim='):
        twins.recipe_contract(source_args(model_backend=twins.TORCHSHIM_BACKEND, hidden_dim=128))
