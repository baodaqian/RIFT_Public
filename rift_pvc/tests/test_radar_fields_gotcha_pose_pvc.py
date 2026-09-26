"""PVC twin of the Radar Fields GOTCHA pose refinement (user decision 2, 2026-09-22): same recipe, poses move."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
for entry in (ROOT, ROOT / "tests"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from rift import radar_fields_gotcha as cuda_gotcha  # noqa: E402
from rift.radar_fields_pose import DISABLED, RELEASE_SE3  # noqa: E402
from rift_pvc import radar_fields_gotcha as pvc_gotcha  # noqa: E402
from test_radar_fields_pvc import _gotcha_config, cpu_backend, needs_reference  # noqa: E402,F401


def test_twin_records_the_same_pose_recipe():
    for config in ({}, {"pose_refinement": DISABLED}):
        pvc, cuda = (m.recipe_from_config(config, .15, 2400) for m in (pvc_gotcha, cuda_gotcha))
        assert pvc.get("pose_refinement") == cuda.get("pose_refinement")
        assert pvc["model_recipe"]["pose_refinement"] == cuda["model_recipe"]["pose_refinement"]


@needs_reference
def test_twin_moves_and_checkpoints_the_sector_poses(tmp_path):
    from test_gotcha_dataset import write_shard, tiny_region
    from rift.gotcha_dataset import GOTCHADataset
    write_shard(tmp_path / "New_Transfer/shards/pass1_hh.npz", 1, "hh", nf=33)
    ds = GOTCHADataset(tmp_path, passes=(1,), polarizations=("hh",), region=tiny_region())
    out = tmp_path / "run"
    assert pvc_gotcha.run_gotcha(dataset=ds, output_dir=out, config=_gotcha_config(), device="cpu",
                                 resume=None)["status"] == "complete"
    saved = torch.load(out / "checkpoint_final.pt", weights_only=False)
    assert saved["recipe"]["pose_refinement"]["name"] == RELEASE_SE3
    assert torch.count_nonzero(saved["pose_model_state_dict"]["hh.adjustment"]) > 0
    with pytest.raises(ValueError):
        pvc_gotcha.run_gotcha(dataset=ds, output_dir=tmp_path / "x", device="cpu",
                              config={**_gotcha_config(), "pose_refinement": DISABLED}, resume=out / "checkpoint_final.pt")
