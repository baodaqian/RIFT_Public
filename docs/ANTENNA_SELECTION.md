# Physical antenna selection

The canonical collection and GOTCHA dispatchers support RIFT, SpINR, Radar
Fields, GeRaF, RadarSplat and Sugavanam–Ertin with one transmitter and one
receiver. Collection defaults are 2400 training views and source Tx 0/Rx 0.
GOTCHA defaults are 1500 sectors and the existing monostatic pair per pulse;
all selected pulses, native frequencies, reference ranges and autofocus remain.

```bash
python train_rift_dataset.py --object a320 \
  --method rift spinr radar_fields geraf radarsplat sugavanam_ertin \
  --num-train 2400 --num-tx 1 --num-rx 1 --dry-run
python train_gotcha_dataset.py \
  --method rift spinr radar_fields geraf radarsplat sugavanam_ertin \
  --num-train 1500 --num-tx 1 --num-rx 1 --dry-run
```

Collection `--num-tx` and `--num-rx` are independent; omitted counts default to
one on the canonical dispatcher. Counts choose the first N source indices.
`--tx-indices 9 2 --rx-indices 3` selects those ordered physical antennas;
explicit counts, if also given, must match. `--num-tx 16 --num-rx 16` restores
all source antennas in their original order. Native GOTCHA accepts only counts
one and indices zero; polarization heads and synthetic-aperture pulses are not
array elements. Antenna selection preserves frequency grids and each selected
method's recipe. Current spatial settings are recorded separately in
[the scene budget](SCENE_BUDGET.md).

`rift/antenna_selection.py` is the shared selector. Public collection loaders
accept `num_tx`, `num_rx`, `tx_indices` and `rx_indices`, or inherit selection
from a run manifest. Low-level APIs without selection retain the source/full
acquisition for compatibility. Direct baseline trainers consume the selected
run manifest emitted by the canonical dispatcher. Original data manifests,
source metadata and NPZ bytes are preserved.
Selected public handles also retain `source_tx_pos` and `source_rx_pos` as
full-array pose metadata, alongside the selected `tx_pos`/`rx_pos`. This does
not grant access to any additional response channels or viewpoint roles.

Selected runs use `<root>/train<N>/<acquisition-label>/<object>/<method>/`;
the label includes dimensions and a digest of ordered source indices. Use the
exact paths printed by the plan. Execution atomically writes `role_manifest.json`
in the object directory; dry runs create no files. Restore the original CLI
root, training count, antenna selection and recipe when resuming.

The sealed contract records selected response shape, original response shape,
ordered antenna indices and a SHA-256 of original per-view antenna geometry
and the frequency grid. The lazy reader checks role permissions before mapping
the stored response and gathers only selected channels. Measurements remain
Tx-outer/Rx-inner; renderers use `[frequency, Rx, Tx]`. A selected pair retains
its actual bistatic positions. Selection precedes normalization, target
construction, occupancy, initialization and fitting. Validation uses the same
pair; reserved-test access still requires explicit opt-in. Original full-MIMO
and other-pair artifacts fail compatibility checks instead of being relabeled.

## Method adaptations

- **RIFT:** selected geometry enters the existing linear renderer/adjoint;
  checkpoint and collection observer identities include selection. Historical
  B787 observer/report gates retain their original full acquisition.
- **SpINR:** selected raw-complex views, train-only normalization, initialization,
  direct-bin rendering and recipe identities use the selected pairs. Pair tiles
  are bounded by available pairs; the signed-real network stays. The selected
  scene budget uses G48 midpoint quadrature on both datasets.
- **Radar Fields:** selected geometry and responses feed the released power
  model. Statistics and recovery/readout bind the selected acquisition. Its
  released 100-profile sampler still samples with replacement, repeating the
  sole pair; this is not 100 independent antennas or proportional fitting savings.
  The sensor orientation comes from the original full-array axes, so selecting
  one element or reversing channel order does not erase or rotate the frame.
- **GeRaF:** a one-pair collection automatically uses `bank_size=1`. The explicit
  `SingleBankGeRaFStage1` subclass returns correctly shaped empty unselected
  tensors while preserving selected-channel gradients and released loss
  normalization. Recipe marker `single_nonempty_bank_v1` separates it from the
  two-bank model; vendor source is unchanged. The selected scene budget uses
  48³ accumulated-only targets and 50000 updates. `geraf_mf48_1t1r.json` records
  the current single-pair option; explicit MF101 protocols remain available.
  GOTCHA retains two banks over all native sector pulses.
- **RadarSplat:** MF-power targets, train-only peak/occupancy and calibration
  records use selected pairs. With one pair, the array-aperture beamwidth
  formula has a zero denominator: the declared `collection_element_hpbw_v1`
  adaptation supplies the simulator's documented 10° element power HPBW to the
  unchanged source filter. The existing deterministic tangent fallback defines
  the target frame when an array span is unavailable. These are acquisition
  approximations, not measured PSF equivalence. The original CUDA model and
  2000 updates remain; the selected `budget48` recipe uses 112000 Gaussians,
  while explicit `upstream` retains 20000.
- **Sugavanam–Ertin:** Stage-1 observations/operators and sample-count budgets
  consume selected pairs. The literal initialization and stage-1 convergence
  gates remain; antenna support does not resolve its existing initialization
  degeneracy or make the method benchmark-ready.

## Readout and verification

RIFT geometry and SpINR readout/quadrature CLIs accept matching `--num-train`,
`--num-tx`, `--num-rx`, `--tx-indices` and `--rx-indices`. GeRaF and Radar Fields
readouts use the saved selected `--role-manifest`; RadarSplat reads its bound
cache/checkpoint. Native GOTCHA readouts retain their saved pulse/sector identity.
Missing reserved-test and native geometry interfaces remain explicitly open.

Bounded tests cover 1t1r, asymmetric/reordered and full selection, exact raw
channel values, role sealing, forward/adjoint/gradients, incompatible identities,
GeRaF single-bank finite gradients and exact interrupted continuation, and tiny
RadarSplat conversion/cache reuse. Metadata-only planning covers six objects ×
six methods and all six GOTCHA backends with response readers blocked.
These checks do not establish original CUDA execution, measured resource savings,
convergence or unchanged resolution. Real-data profiling and experiments remain
manager-owned; no execution or publication is authorized by this implementation.
