"""SpINR recipe identity states the measurement channels one update actually groups."""
import pytest

import train_spinr_style as cuda
import train_spinr_style_pvc as pvc
from rift.antenna_selection import selection

ONE_PAIR = selection(1, 1, None, None)


@pytest.mark.parametrize('lane', [cuda, pvc])
@pytest.mark.parametrize('recipe', ['budget48-direct', 'budget48-direct-1500', 'paper-v1-direct'])
def test_one_pair_identity_states_four_channels_and_old_checkpoints_still_match(lane, recipe):
    identity = lane._recipe_identity(recipe, 2400, ONE_PAIR)
    assert lane.PAPER_BATCH_TEXT not in identity['fidelity']['specified']
    assert any(text.startswith('four whole views per update = 4 measurement channels')
               for text in identity['fidelity']['benchmark_settings'])
    declared = lane._declared_recipe_identity(recipe, 2400, ONE_PAIR)
    assert lane._legacy_batch_metadata(identity) == declared      # what C1 checkpoints carry
    for saved in (identity, declared):
        with pytest.raises(ValueError) as error:                  # passes the recipe gate, fails later
            lane._validate_resume_checkpoint_structure(
                {'format': lane.CHECKPOINT_FORMAT, 'spinr_style_recipe': saved}, recipe_identity=identity)
        assert 'scientific recipe' not in str(error.value)
    changed = lane._declared_recipe_identity(recipe, 2400, ONE_PAIR)
    changed['optimizer']['lr'] = 1e-3
    with pytest.raises(ValueError, match='scientific recipe'):
        lane._validate_resume_checkpoint_structure(
            {'format': lane.CHECKPOINT_FORMAT, 'spinr_style_recipe': changed}, recipe_identity=identity)


@pytest.mark.parametrize('lane', [cuda, pvc])
def test_full_array_identity_is_unchanged(lane):
    for recipe in ('budget48-direct', 'paper-v1-direct', 'legacy-midpoint'):
        assert lane._recipe_identity(recipe, 2400) == lane._declared_recipe_identity(recipe, 2400)
    assert lane.PAPER_BATCH_TEXT in lane._recipe_identity('paper-v1-direct', 2400)['fidelity']['specified']
