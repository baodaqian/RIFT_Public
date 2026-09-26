# GeRaF-SENS source attribution — PVC (Intel XPU) twins

Copyright (c) 2026 Laboratory of Sensing and Networking Systems, EPFL.
Licensed under the included PolyForm Noncommercial License 1.0.0.

Source: https://github.com/VictorLlu/GeRaF-SENS/tree/38266cb6e194e2f3dcbead614069a7281ffd21a5

These two files are PVC twins of `rift/vendor/geraf_sens/`, created for the
Intel Data Center GPU Max port (Package B of `RIFT_PVC_Adaptation.md`). The CUDA
originals under `rift/vendor/geraf_sens/` are unchanged and remain the reference
for the CUDA pipeline; the full attribution list, including every other vendored
definition, is in `rift/vendor/geraf_sens/NOTICE.md`.

## Files and the only differences from the CUDA copies

- `rf_rendering.py` — shared helpers, `GeRaFStageBase` and the complete
  `GeRaFStage1` including its loss and antenna-bank lifecycle, from
  `geraf/models/rendering/rf_rendering.py`. Three `torch.amp.autocast("cuda",
  dtype=torch.float16)` context managers (in `_ensure_cached_adc`,
  `_calibrate_transmission` and `GeRaFStage1.loss`) become
  `rift_pvc.geraf_autocast.autocast_fp16()`, which routes through the accelerator
  shim with float16 enabled on XPU/CUDA and disabled on CPU. Model bodies,
  constants and the `rt_backend`/`mf_backend` default strings `"cuda"` are
  unchanged: those strings are
  inert here, because `rift/geraf_source_ops.py` accepts the `backend` keyword
  and always dispatches to the pure-torch `radar_cfg['native']` acquisition, and
  `rift_pvc/geraf_source.py::build_model` passes `'native'` regardless. No CUDA
  extension is compiled or loaded; the `*_reference.cu` files next to the CUDA
  copies are documentation.
- `train_reference.py` — the preserved upstream runner. It imports the upstream
  `geraf` package, which this repository does not vendor, so — exactly like the
  CUDA copy — it is reference material and is never imported or executed by this
  project. `choose_device`, the seeding helper, the `DataLoader` `pin_memory`
  flag and the `to_device` `non_blocking` flag go through
  `rift_pvc.accelerator`, so the reference shows what a PVC runner would do.

The remaining vendored modules (`sdf_network.py`, `power_network.py`,
`sample.py`, `loading.py`, `scheduler.py`, `base.py`, `field.py`, `registry.py`,
`transform.py`, `stage1_config_reference.py`) are device-free and are imported
unchanged from `rift/vendor/geraf_sens/`; they are not duplicated here.

Per the repository policy in `AGENTS.md`, provenance for these files is the
upstream repository URL and the selected commit above. No content-hash
verification is performed on them, and none should be added. The historical
`source_manifest.json` beside the CUDA copies stays documentation only.
