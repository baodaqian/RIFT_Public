# GOTCHA native frequency selection

`--frequency-stride 2` applies the same source-bin selection to **RIFT, SpINR,
Radar Fields, GeRaF, RadarSplat and Sugavanam–Ertin**, in train, validation and
reserved-test metadata. Combine it with the shared [fixed pulse cap](GOTCHA_PULSE_SELECTION.md):

```bash
python train_gotcha_dataset.py \
  --method rift spinr radar_fields geraf radarsplat sugavanam_ertin \
  --pulses-per-sector 16 --frequency-stride 2 --dry-run
```

The public API is `GOTCHADataset(frequency_stride=2, pulses_per_sector=16)`.
Defaults `frequency_stride=1, pulses_per_sector=0` preserve the original
acquisition and identities. These options concern native GOTCHA; the simulated
RIFT collection retains its separate view/antenna controls. Implementation does
not authorize training, conversion or scheduler actions.

## Acquisition and evaluation contract

Stride 2 retains source indices `0, 2, 4, ...` and the last source index if
missing, preserving both bandwidth endpoints. Actual frequency values, native
pulse IDs, geometry, phase reference and source-owned autofocus remain intact.
No averaging, uniform-grid substitution or archive modification occurs.
Readers retain full source `.shape` and `.frequencies_hz`; adapters use
`.frequencies_for_role(role)` and `.frequency_indices_for_role(role)` for the
declared selection. All three roles use the same rule. Test metadata selection
does **not** open test payloads: training/preparation readers still reject them.

Normalization and training targets use selected TRAIN responses only. Validation
targets and metrics use selected validation responses. Existing unprojected
native-complex metric keys describe all samples of this declared acquisition;
they do not imply the original full-density archive when selection is enabled.
Any full-density diagnostic needs a separate declared evaluation. Reserved-test
reporting still requires its explicit evaluation path/authorization.

This changes the acquisition and training objective; it is not an unbiased
estimate of the original full-frequency loss. The comparison uses matching
sampling rules across methods and roles, without changing model/update budgets.

| Method | Frequency-dependent behavior |
| --- | --- |
| RIFT | Builds the ROI projector from exact selected frequencies; projected TRAIN power defines normalization. Cropping an existing full-frequency projection would be incorrect. |
| SpINR | Uses selected native values in its exact DFT kernel, TRAIN normalization/initialization and resume sample counts. |
| Radar Fields | Rebuilds matched-range power with the selected-frequency mean. Released 100-profile sampling and fitting budget remain. |
| GeRaF | Geometry, responses, accumulated TRAIN MF and lazy role targets use selected frequencies. Released antenna-mean/frequency-**sum** normalization remains; there is no stride compensation. |
| RadarSplat | Converts role targets from selected pulses/frequencies, preserving frequency and pulse means. Gaussian fitting retains its budget. |
| Sugavanam–Ertin | Stage-1 operators and residual sample counts use the selected acquisition. Initialization, solver and convergence gates remain. |

## Identity, recovery and qualification

`contract.frequency_selection` uses `gotcha_role_native_frequency_selection_v2`:
role rules, exact source-bin indices, source/selected frequency hashes, endpoint
coverage, normalization and projector policy. Cache/checkpoint gates bind this
contract. Output paths add `frequency_stride2_all_roles_v2`, after any
`pulse_subset<N>_all_roles_v2` directory. Use the original output root and same
controls for recovery. Changed selections and the superseded training-only
prototype require separate runs. GeRaF and RadarSplat readouts reconstruct both
controls from saved contracts before their normal identity/response gates.

Metadata preflight checks the largest guarded cube-delay support over selected
train/validation/test pulses in each shard using the unchanged half-Rayleigh
projector, two guards and SVD cutoff `1e-10`. Basis columns must be fewer than
selected frequencies; the guarded interval must fit the conservative alias
period. Stride 4 is rejected, not treated as a qualified extension.

Default eight-pass Camry HH with cap 16/stride 2 uses 213–218 frequencies per
pulse. The maximum guarded basis has 152 columns and retained rank 97–98;
support/rank/alias gates pass without reading responses. Selected sample counts
are 5,169,008 TRAIN, 1,516,240 validation and 1,516,240 reserved test, versus
76,151,564 / 22,338,864 / 22,339,226 without either selection. These are acquisition
counts, not runtime or quality measurements.

Owner tests cover native-bin physics, every baseline adapter, selected-role
metrics/targets, sealed test/excluded reads, normalization and artifact gates,
saved-selection readouts, and exact interrupted RIFT/SpINR recovery. Metadata
evidence is in `experiment_state/gotcha_frequency_selection_20260921/`.
GPU cost, memory and convergence remain manager-owned qualification gates.
