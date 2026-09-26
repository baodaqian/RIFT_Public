"""PVC twin: the GeRaF light_power warm start runs through rift_pvc's training lane."""
import json

from rift_pvc import geraf_source_training as runtime
from tests.test_geraf_light_power_start import WARM
from tests.test_geraf_source import TinyData


def test_pvc_training_records_the_warm_start(tmp_path, monkeypatch):
    monkeypatch.setenv('RIFT_ACCELERATOR', 'cpu')
    assert runtime.train(data=TinyData(), output_dir=tmp_path / 'warm', config=WARM, device='cpu')['status'] == 'complete'
    assert json.loads((tmp_path / 'warm/light_power_warm_start.json').read_text())['scalar']['light_power'] != 0.0
