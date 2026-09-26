# Fixed GOTCHA pulse selection for training and evaluation

`train_gotcha_dataset.py --pulses-per-sector N` applies the same fixed native
acquisition in train, validation and test roles to RIFT, SpINR, Radar Fields, GeRaF, RadarSplat and
Sugavanam–Ertin. The public API is `GOTCHADataset(pulses_per_sector=N)`.
The default `0` retains all pulses and the historical dataset identity.
Positive N retains up to N pulses in each pass-sector/channel in all three roles.

```bash
python train_gotcha_dataset.py \
  --method rift spinr radar_fields geraf radarsplat sugavanam_ertin \
  --pulses-per-sector 16 --dry-run
```

This is an opt-in acquisition change. The fixed subset defines a smaller
training objective and evaluation acquisition; it is not per-update resampling
or an unbiased estimator of the complete native loss. Compare all six methods
using the same cap, split, source and frequency selection. Equal reconstruction
quality remains to be established. The implemented `--frequency-stride {1,2}`
control composes with this cap; see [frequency selection](GOTCHA_FREQUENCY_SELECTION.md).

## Selection and identity

`rift/gotcha_pulse_sampling.py` assigns each source pulse a BLAKE2b 64-bit
priority from `(42, pass_id, sector_id, pulse_index)` using little-endian signed
64-bit integers and personalization `RIFTpulsev1`. It retains the lowest N
priorities without replacement and restores native pulse order. Increasing N
gives nested subsets. The hash does not depend on model, optimizer RNG, run
directory, polarization or response values. Channels with identical pulse-ID
inventories therefore share the same mask. Native channel inventories remain
authoritative when channels contain different pulse IDs.

Selection happens after source metadata validation and before response access.
Both `sector_rows` and `row_roles` reflect the subset, so direct baseline shard
access and `dataset.observations()` agree. Unselected rows in every role are excluded
and direct reads fail before mapping responses. Original response shape,
frequency vector, source headers, row/pulse IDs, positions, ranges and archive
bytes remain intact. Source role/sector metadata are retained separately on the
reader. HH/VV autofocus still belongs to the ingress and is applied once.

Validation and test use the same fixed cap and priority rule as training.
Test responses stay sealed in the training adapter; their selected row metadata
are ready for the separate explicitly authorized evaluation path. No test
response is read during selection or preparation. Selecting training sectors
with `--num-train` happens first; holdout sector IDs stay fixed and the pulse cap
applies inside each role's sectors. The simulated RIFT collection has one source chirp
per view and uses its separate view/antenna controls.

For N > 0, `contract.training_pulse_selection` records schema
`gotcha_fixed_role_pulses_v2`, cap, seed, priority policy, all three roles,
normalization policy and per-shard/per-role counts/SHA-256 of selected original
rows and pulse IDs. The historical property/key name is retained for API
compatibility; its explicit role list includes validation and test.
This changes the dataset identity and every dependent cache/checkpoint identity.
The CLI adds `pulse_subset<N>_all_roles_v2` below the dataset identity directory. Use paths
from the plan and the same cap/output root when resuming. Changed caps require
a fresh run, even if a cap exceeds every sector's native pulse count.

RadarSplat's `cache_from_run` and GeRaF's source validation readout restore the
cap from the saved authoritative acquisition contract, then apply the normal
source/recipe identity gates before response access. Legacy saved contracts
without this field reconstruct all pulses. The earlier TRAIN-only prototype
schema is rejected; it cannot be silently reinterpreted as matched evaluation.

## Effect on each method

| Method | Work using the selected acquisition |
| --- | --- |
| RIFT | Normalization, coherent loss and adaptive gradient statistics use selected pulses. One optimizer update still covers one pass-sector. The sector mean divides by its actual selected count; refinement records actual selected observations. |
| SpINR | Normalization, scale initialization, flattened pulse batches and recovery exposure counts use selected native rows. Fewer pulses produce fewer batches per epoch at the existing 1024-pulse budget. |
| Radar Fields | Normalization and eligible frame profiles use the subset. The released sampler still draws 100 profiles with replacement per training frame; its model compute does not fall proportionally to the pulse cap. |
| GeRaF | Native channel geometry/responses, training MF accumulation and lazy targets use selected rows; released network/update settings remain. |
| RadarSplat | Selected geometry and responses form fixed training and validation MF-power targets and calibration; normalization remains TRAIN-only. Gaussian fitting retains its model/update budget, so the main savings are in conversion and evaluation preparation. |
| Sugavanam–Ertin | Selected pulses determine sub-aperture membership, Stage-1 operators, normalization and residual sample counts. The constrained solver receives a fixed acquisition; initialization/convergence gates remain. |

All six score the complete declared selected validation role and bind their
preparation artifacts to the selected acquisition. Normalization never uses
validation or test responses. Full-native diagnostics, if requested separately,
must be identified as a different evaluation acquisition. No permanent response averaging, decimation of source
archives, new renderer approximation or surface prior is introduced.

## Validation boundary

Synthetic tests in `tests/test_gotcha_pulse_sampling.py` exercise ragged sectors,
native ordering/geometry/autofocus, shared channel masks, sealed excluded rows,
six-method metadata plans, each baseline's actual acquisition adapter, selected
validation metrics, per-role inventory hashes, saved-cap
readouts, cache/checkpoint mismatch rejection and exact interrupted adaptive
RIFT recovery including optimizer/refinement state.

Response-blocked native metadata preflight for the default 1500 HH train sectors
gives 177605 pulses with cap 0 and 24000 with cap 16 (about 7.4× fewer training
pulses). Cap 16 also selects 7040 validation and 7040 sealed test pulses over
440 sectors each, versus 52100/52101 at cap 0. These are acquisition
counts, not measured end-to-end speedups. GPU fit, memory, convergence and
selected-role validation quality require manager-owned qualification; no production
training or conversion is authorized by these implementation checks.
