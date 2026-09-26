"""Native GOTCHA binding of the same source GeRaFStage1 used on RIFT."""
from rift.geraf_source_data import GOTCHASourceData
from rift.geraf_source_training import train


def run_gotcha(*, dataset, output_dir, config, device, resume):
    return train(data=GOTCHASourceData(dataset), output_dir=output_dir,
                 config=config, device=device, resume=resume)
