"""Native GOTCHA binding of the same source GeRaFStage1, PVC (Intel XPU) twin.

Copy of ``rift/geraf_gotcha.py`` bound to the PVC training module. The data
adapter is the device-free ``GOTCHASourceData`` from ``rift/``; ``run_gotcha``
keeps its exact signature so ``train_gotcha_dataset_pvc.py`` binds it the same
way the CUDA frontend binds the original.
"""
from rift.geraf_source_data import GOTCHASourceData
from rift_pvc.geraf_source_training import train


def run_gotcha(*, dataset, output_dir, config, device, resume):
    return train(data=GOTCHASourceData(dataset), output_dir=output_dir,
                 config=config, device=device, resume=resume)
