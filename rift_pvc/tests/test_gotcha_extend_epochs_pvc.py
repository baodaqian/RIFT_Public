"""RIFT-on-GOTCHA tuning campaign A55 item 4: extending a completed run past its recipe's epochs."""
import importlib.util
import json

import pytest
import torch

import train_gotcha_dataset_pvc as cli
from rift.gotcha_dataset import GOTCHADataset
from rift_pvc import gotcha_training as pvc
from rift_pvc.gotcha_unit_split import apply_unit_split
from rift_pvc.tests.test_gotcha_densify_pvc import ARGV, BOX, root  # noqa: F401  (fixture)
from tests.test_gotcha_dataset import tiny_region

spec = importlib.util.spec_from_file_location('extend', 'scripts_pvc/gotcha_extend_epochs.py')
extend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extend)

DENSE = ['--passes', '1', '2', '--unit-split-stride', '4', '--unit-split-heldout-pass', '2',
         '--densify-epochs', '1', '--densify-max-active', '600', '--densify-lr-rule', 'fixed',
         '--densify-lr-factor', '0.5', '--train-eval-every', '1', '--lr', '3e-4', '--pos-lr', '3e-4',
         '--loss-domain', 'full_native']


def dataset(root):
    ds = GOTCHADataset(root, passes=(1, 2), region=tiny_region(), pulses_per_sector=3, num_train=2)
    apply_unit_split(ds, stride=4, heldout_pass=2, heldout_fraction=0.5)
    return ds


def recipe(epochs, extra=()):
    assert ARGV[0] == '--epochs'
    return pvc.recipe_from_args(cli.parse_args(['--epochs', str(epochs)] + ARGV[2:] + BOX + DENSE + list(extra)), 'rift')


def run(root, name, epochs, resume=None):
    """Train into ROOT/NAME/region/rift_full_native, as the CLI nests a run below its output root."""
    out = root / name / 'region' / 'rift_full_native'
    out.parent.mkdir(parents=True, exist_ok=True)
    (root / name / 'train.log').touch()
    return pvc.train(dataset(root), 'rift', recipe(epochs), out, device='cpu', resume=resume), out


def test_restart_epochs_follow_the_warm_restart_schedule():
    assert extend.restart_epochs(10, 2, 80) == [10, 30, 70]
    assert extend.restart_epochs(10, 1, 35) == [10, 20, 30]


def test_extension_resumes_bit_identically_to_a_run_declared_longer(root):
    status, short = run(root, 'short', 2)
    assert status['completed_epochs'] == 2
    extend.main([str(short), str(root / 'extended'), '--epochs', '3', '--note', 'test'])
    target = root / 'extended' / 'region' / 'rift_full_native'
    saved = torch.load(target / 'checkpoint_latest.pt', weights_only=False)
    source = torch.load(short / 'checkpoint_final.pt', weights_only=False)
    assert saved['recipe'] == dict(source['recipe'], epochs=3) and saved['epoch'] == 2
    status, _ = run(root, 'extended', 3, resume=target / 'checkpoint_latest.pt')
    assert status['completed_epochs'] == 3
    _, full = run(root, 'full', 3)
    a = torch.load(target / 'checkpoint_final.pt', weights_only=False)
    b = torch.load(full / 'checkpoint_final.pt', weights_only=False)
    assert a['model_state_dict'].keys() == b['model_state_dict'].keys()
    for k, v in a['model_state_dict'].items():
        assert torch.equal(v, b['model_state_dict'][k]), k
    ha = json.loads((target / 'history.json').read_text())
    hb = json.loads((full / 'history.json').read_text())
    record = ha[1].pop('extension')
    assert record['from_epochs'] == 2 and record['to_epochs'] == 3 and record['restarts_crossed'] == []
    assert ha == hb


def test_extension_refusals(root):
    _, short = run(root, 'short', 2)
    for argv, message in ((['--epochs', '2'], 'must exceed'), (['--epochs', '11'], 'restart after epoch 10')):
        with pytest.raises(SystemExit, match=message):
            extend.main([str(short), str(root / 'refused'), *argv])
    extend.main([str(short), str(root / 'past'), '--epochs', '11', '--past-restart'])
    record = json.loads((root / 'past/region/rift_full_native/extension.json').read_text())
    assert record['restarts_crossed'] == [10] and record['refinement_events_crossed'] == [10]
    assert record['unguarded_acknowledged'] == ['restart after epoch 10', 'refinement event after epoch 10']
    with pytest.raises(SystemExit, match='not empty'):
        extend.main([str(short), str(root / 'past'), '--epochs', '4'])
    # A mid-run checkpoint is refused.
    mid = torch.load(short / 'checkpoint_final.pt', weights_only=False)
    mid['epoch'] = 1
    torch.save(mid, short / 'checkpoint_mid.pt')
    with pytest.raises(SystemExit, match='not a completed run'):
        extend.main([str(short), str(root / 'mid'), '--epochs', '4', '--checkpoint', 'checkpoint_mid.pt'])


def test_a_densify_event_at_the_old_end_runs_on_resume_as_declared(root):
    # Densify after epoch 1 is skipped by a 1-epoch run (its last epoch); extended to 3, the resume branch
    # runs it before epoch 2, exactly where the declared 3-epoch run does (A62).
    _, short = run(root, 'short', 1)
    assert 'densify' not in json.loads((short / 'history.json').read_text())[0]['optimizer']
    extend.main([str(short), str(root / 'extended'), '--epochs', '3'])
    target = root / 'extended' / 'region' / 'rift_full_native'
    assert json.loads((target / 'extension.json').read_text())['densify_events_ahead'] == [1]
    run(root, 'extended', 3, resume=target / 'checkpoint_latest.pt')
    _, full = run(root, 'full', 3)
    a = torch.load(target / 'checkpoint_final.pt', weights_only=False)
    b = torch.load(full / 'checkpoint_final.pt', weights_only=False)
    for k, v in a['model_state_dict'].items():
        assert torch.equal(v, b['model_state_dict'][k]), k
    ha = json.loads((target / 'history.json').read_text())
    ha[0].pop('extension')
    assert ha == json.loads((full / 'history.json').read_text())
