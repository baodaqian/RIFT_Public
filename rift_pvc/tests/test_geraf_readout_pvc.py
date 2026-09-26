"""The public PVC reader must reproduce training validation on the same card."""
import pytest

from scripts_pvc import eval_geraf_source_pvc as reader
from rift_pvc import geraf_source_training as runtime
from rift_pvc.tests.test_geraf_xpu import device, DeviceTinyData
from tests.test_geraf_source import SMALL


def test_public_readout_matches_saved_validation(device, tmp_path, monkeypatch):
    data = DeviceTinyData(device)
    output = tmp_path/"fit"
    result = runtime.train(data=data, output_dir=output, config=SMALL, device=device)
    assert result["status"] == "complete"
    monkeypatch.setattr(reader, "RIFTSourceData", lambda *args: data)
    monkeypatch.setattr("rift.rift_dataset.resolve_object_inputs", lambda **kwargs: ("npz", "roles"))
    report = reader.main(["--dataset", "rift", "--object", "b787",
                          "--checkpoint", str(output/"checkpoint_best.pth.tar"),
                          "--cache-root", str(output/"source_targets"),
                          "--output", str(tmp_path/"readout.json")])
    assert report["device_backend"] == device.type
    assert report["autocast_dtype"] == (None if device.type == "cpu" else "float16")
    assert report["role"] == "validation"
    assert report["metrics"]["mf_magnitude_mse"] == report["selection_record"]["mf_magnitude_mse"]
    assert all(role in ("train", "validation") for role, _ in data.reads)
    with pytest.raises(FileExistsError):
        reader.main(["--dataset", "rift", "--object", "b787", "--checkpoint", "unused",
                     "--cache-root", "unused", "--output", str(tmp_path/"readout.json")])
