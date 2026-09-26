"""Focused device audit; run explicitly with pytest, not a production trainer.

The historical Stage-2 test expresses the documented CPU-generator contract.
It intercepts construction so a defective XPU generator cannot hang the job.
The GOTCHA test checks actual optimizer/refinement/RNG recovery on the card.
"""
import signal
from unittest.mock import patch

import pytest
import torch

from rift_pvc import accelerator
from rift_pvc import gotcha_training
from tests.test_gotcha_frequency_selection import native_data, dataset  # noqa: F401
from tests.test_spinr_gotcha import assert_tree_equal
import train_gotcha_dataset_pvc as frontend


@pytest.mark.skipif(accelerator.backend() != "xpu", reason="XPU-specific historical wrapper")
def test_historical_stage2_constructs_cpu_generator():
    import train_sugavanam_ertin_stage2 as original
    import train_sugavanam_ertin_stage2_pvc as pvc
    real_generator = torch.Generator
    requested = []

    def intercept(*args, **kwargs):
        requested.append(str(kwargs.get("device", args[0] if args else "cpu")))
        # Never create an XPU generator, even when the wrapper is defective.
        return real_generator(device="cpu")

    try:
        pvc.install()
        with patch.object(torch, "Generator", side_effect=intercept):
            # Exact construction expression used by original._run (line 1325).
            original.torch.Generator(device=accelerator.device())
    finally:
        pvc.uninstall()
    assert requested == ["cpu"], f"Historical Stage-2 still requests {requested}"


@pytest.mark.skipif(not accelerator.is_available(), reason="requires allocated GPU")
def test_gotcha_rift_device_interrupted_resume(native_data, monkeypatch):
    ds = dataset(native_data)
    original_views = ds.viewpoints
    monkeypatch.setattr(ds, "viewpoints", lambda role:
                        original_views(role)[:1] if role == "validation" else original_views(role))
    args = frontend.parse_args([
        "--epochs", "2", "--granularity", "2", "--max-points", "15",
        "--sh-degree", "1", "--refine-every", "2", "--probe-every", "1",
    ])
    recipe = gotcha_training.recipe_from_args(args, "rift")

    def run(output, **kwargs):
        return gotcha_training.train(ds, "rift", recipe, output,
                                     device=accelerator.device(), **kwargs)

    full_root, resumed_root = native_data / "full", native_data / "resumed"
    assert run(full_root)["status"] == "complete"
    real_step = torch.optim.AdamW.step

    def interrupt(optimizer, *args, **kwargs):
        result = real_step(optimizer, *args, **kwargs)
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        return result

    with monkeypatch.context() as local:
        local.setattr(torch.optim.AdamW, "step", interrupt)
        assert run(resumed_root)["status"] == "interrupted"
    checkpoint = resumed_root / "checkpoint_latest.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["cursor"] == 1
    assert saved["rng_state"]["accelerator_backend"] == accelerator.backend()
    assert len(saved["rng_state"]["accelerator"]) == accelerator.device_count()
    assert run(resumed_root, resume=checkpoint)["status"] == "complete"
    full = torch.load(full_root / "checkpoint_final.pt", map_location="cpu", weights_only=False)
    resumed = torch.load(resumed_root / "checkpoint_final.pt", map_location="cpu", weights_only=False)
    for key in ("model_state_dict", "optimizer_state_dict", "history", "rng_state",
                "training_statistics", "epoch", "cursor", "updates", "order"):
        assert_tree_equal(full[key], resumed[key])
