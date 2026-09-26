# Bounded PVC smoke configurations

`protocols/` holds the frozen production recipes and is never edited. The
current GeRaF smoke uses `protocols/geraf_mf48_1t1r.json` unchanged: 50000 steps,
the original cosine clock and original checkpoint/validation/log intervals.
`scripts_pvc/smoke_geraf_pvc.py` sends SIGTERM at the 100-update checkpoint
cadence, checks the partial checkpoint, resumes its explicit path and requires
at least 100 additional updates. Its watchdog, unexpected exits, absent or
invalid recovery state, and XPU fallback warnings are failures. The resulting
smoke report means bounded execution/recovery passed, not production completion.
Use a fresh output root; the launcher defaults to `geraf_production_clock_smoke`.

- Historical only: `geraf_mf48_1t1r_pvcsmoke.json` — `protocols/geraf_mf48_1t1r.json`
  (`mf_grid` 48, `bank_size` 1: the 1t1r production target protocol) with
  `steps` 24, `checkpoint_every` 8, `validation_every` 24, `log_every` 1.
  Production is `steps` 50000, `checkpoint_every` 100, `validation_every` 1000,
  `log_every` 10. The historical file compresses the cosine schedule to 24
  steps too; those runs are not prefixes of the production trajectory. Retain
  this configuration only to interpret/recover their original checkpoints.
  The current smoke rejects it instead of silently altering the schedule.

PVC source readout: `python scripts_pvc/eval_geraf_source_pvc.py` accepts the
original evaluator's arguments, defaults to the active accelerator, and uses
the training runtime's fp16 autocast on XPU/CUDA. Reports identify device and
autocast precision. CPU readout remains an explicitly labeled fp32 diagnostic.

PVC acceptance tests use `python -m pytest -p rift_pvc.pytest_policy ...`.
The plugin deselects only the inherited vendored-source hash assertion while
leaving cache/checkpoint integrity tests active. Source provenance is checked
by URL/commit, required license files, imports and executable APIs. The
protected CUDA test files are unchanged.
